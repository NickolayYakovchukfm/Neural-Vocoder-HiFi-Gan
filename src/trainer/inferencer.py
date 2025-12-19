import torch
import torchaudio
from torch import autocast
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        save_path,
        text,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            save_path (str): path to save model predictions and other
                information.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config

        self.device = device

        self.model = model
        self.t2m = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_tacotron2",
            model_math="fp16",
            pretrained=False,
        )
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.09.0/files/nvidia_tacotron2pyt_fp16_20190427",
            map_location=self.device,
        )
        state_dict = {
            key.replace("module.", ""): value
            for key, value in checkpoint["state_dict"].items()
        }
        self.t2m.load_state_dict(state_dict)
        self.t2m.to(device).eval()
        self.t2m._modules["decoder"].max_decoder_steps = 5000
        self.utils = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils"
        )

        # path definition
        self.save_path = save_path

        self.text = text

        if not skip_model_load:
            # init model
            self._from_pretrained(config.get("from_pretrained"))
        self.is_amp = config.get("is_amp", True)

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        self._inference_part()

    def process_batch(self, batch_idx, batch):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        with autocast(
            device_type=self.device, enabled=self.is_amp, dtype=torch.float16
        ):
            sequences, lengths = self.utils.prepare_input_sequence(
                [batch["text"]], self.device == "cpu"
            )
            mel = self.t2m.infer(sequences, lengths)[0]
            outputs = self.model(spectrogram=mel)

        if self.save_path is not None:
            torchaudio.save(
                self.save_path / f"{batch['name']}.wav",
                outputs["predict"].to(torch.float32).cpu(),
                22050,
            )

        return batch

    def _inference_part(self):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(self.text),
                total=len(self.text),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                )
