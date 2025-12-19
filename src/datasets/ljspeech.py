import csv
import json
import random

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class LJSpeechDataset(BaseDataset):
    def __init__(
        self,
        data_dir=None,
        segment_size=8192,
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
        *args,
        **kwargs,
    ):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "LJSpeech-1.1"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True,
        )

        index = self._get_or_load_index()

        super().__init__(index, limit, shuffle_index, instance_transforms)

    def _get_or_load_index(self):
        index_path = self._data_dir / "index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []
        with open(self._data_dir / "metadata.csv", "r") as csv_file:
            csv_file_iter = csv.reader(csv_file, delimiter="|", quotechar="|")
            for file_id, _, _ in csv_file_iter:
                path = str(self._data_dir / "wavs" / f"{file_id}.wav")
                t_info = torchaudio.info(str(path))
                length = t_info.num_frames / t_info.sample_rate
                index.append({"path": path, "label": length})
        return index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio = self.load_object(audio_path)

        if audio.shape[1] >= self.segment_size:
            max_audio_start = audio.shape[1] - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start : audio_start + self.segment_size]
        else:
            audio = torch.nn.functional.pad(
                audio, (0, self.segment_size - audio.shape[1]), "constant"
            )

        mel = self.mel_spectrogram(audio)

        instance_data = {
            "audio": audio,
            "spectrogram": mel.squeeze(0),
            "audio_path": audio_path,
        }
        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def load_object(self, path):
        audio, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
        return audio