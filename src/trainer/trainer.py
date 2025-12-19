from pathlib import Path

import pandas as pd
import torch
from torch import autocast

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.loss.hifigan_loss import FeatureMatchingLoss, MelSpectrogramLoss


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fm_loss = FeatureMatchingLoss().to(self.device)
        self.mel_loss = MelSpectrogramLoss().to(self.device)
        self.fm_lambda = 2
        self.mel_lambda = 45

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["train"]
        self.optimizer["g_optimizer"].zero_grad()
        self.optimizer["d_optimizer"].zero_grad()

        with autocast(
            device_type=self.device, enabled=self.is_amp, dtype=torch.bfloat16
        ):
            outputs = self.model.Generator(**batch)
            batch.update(outputs)
            mpd_out_real, mpd_out_fake, mpd_feats_real, mpd_feats_fake = self.model.MultiPeriodDiscriminator(
                batch["audio"].unsqueeze(1), batch["predict"].detach()
            )
            msd_out_real, msd_out_fake, msd_feats_real, msd_feats_fake = self.model.MultiScaleDiscriminator(
                batch["audio"].unsqueeze(1), batch["predict"].detach()
            )
            batch.update({
                "mpd_out_real": mpd_out_real,
                "mpd_out_fake": mpd_out_fake,
                "mpd_feats_real": mpd_feats_real,
                "mpd_feats_fake": mpd_feats_fake,
                "msd_out_real": msd_out_real,
                "msd_out_fake": msd_out_fake,
                "msd_feats_real": msd_feats_real,
                "msd_feats_fake": msd_feats_fake,
            })

            d_loss = self.criterion["d_loss"](
                [batch["mpd_out_real"], batch["msd_out_real"]],
                [batch["mpd_out_fake"], batch["msd_out_fake"]]
            )
            batch["d_loss"] = d_loss

        batch["d_loss"].backward()
        self._clip_grad_norm("d")
        self.optimizer["d_optimizer"].step()
        metrics.update("d_grad_norm", self._get_grad_norm("d"))

        self.optimizer["g_optimizer"].zero_grad()

        with autocast(
            device_type=self.device, enabled=self.is_amp, dtype=torch.bfloat16
        ):
            outputs = self.model.Generator(**batch)
            batch.update(outputs)
            
            batch["spectrogram_predict"] = self.batch_transforms.get("train")[
                "spectrogram"
            ](batch["predict"])
            
            mpd_out_real, mpd_out_fake, mpd_feats_real, mpd_feats_fake = self.model.MultiPeriodDiscriminator(
                batch["audio"].unsqueeze(1), batch["predict"]
            )
            msd_out_real, msd_out_fake, msd_feats_real, msd_feats_fake = self.model.MultiScaleDiscriminator(
                batch["audio"].unsqueeze(1), batch["predict"]
            )
            batch.update({
                "mpd_out_real": mpd_out_real,
                "mpd_out_fake": mpd_out_fake,
                "mpd_feats_real": mpd_feats_real,
                "mpd_feats_fake": mpd_feats_fake,
                "msd_out_real": msd_out_real,
                "msd_out_fake": msd_out_fake,
                "msd_feats_real": msd_feats_real,
                "msd_feats_fake": msd_feats_fake,
            })

            loss_g_adv = self.criterion["g_loss"](
                [batch["mpd_out_fake"], batch["msd_out_fake"]]
            )
            loss_fm_mpd = self.fm_loss(batch["mpd_feats_real"], batch["mpd_feats_fake"])
            loss_fm_msd = self.fm_loss(batch["msd_feats_real"], batch["msd_feats_fake"])
            loss_fm = loss_fm_mpd + loss_fm_msd
            loss_mel = self.mel_loss(batch["predict"], batch["audio"].unsqueeze(1))
            
            g_loss = loss_g_adv + self.fm_lambda * loss_fm + self.mel_lambda * loss_mel
            batch["g_loss"] = g_loss

        batch["g_loss"].backward()
        self._clip_grad_norm("g")
        self.optimizer["g_optimizer"].step()
        metrics.update("g_grad_norm", self._get_grad_norm("g"))

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        self.log_predictions(**batch)

    def log_predictions(self, audio, predict, audio_path, **batch):
        for i in range(min(5, audio.shape[0])):
            audio_name = Path(audio_path[i]).stem
            self.writer.add_audio(
                f"audio_target_{audio_name}",
                audio[i],
                sample_rate=22050
            )
            self.writer.add_audio(
                f"audio_predicted_{audio_name}",
                predict[i].squeeze(0),
                sample_rate=22050
            )
