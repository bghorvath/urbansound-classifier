import yaml
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, transform, device) -> None:
        self.device = device
        self.annotations = pd.read_csv("data/UrbanSound8K/UrbanSound8K.csv")
        self.transform = transform.to(self.device)
        params = yaml.safe_load(open("params.yaml"))
        self.target_sr = params["transform"]["params"]["sample_rate"]
        self.duration = params["transform"]["params"]["duration"]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int):
        audio_path = self._get_audio_path(index)
        label = self._get_label(index)
        signal, sr = torchaudio.load(audio_path).to(self.device)
        signal = self._resample(signal, sr)
        signal = self._mix_down(signal)
        signal = self._cut(signal)
        signal = self.transform(signal)
        return signal, label
    
    def _cut(self, signal):
        length = self.target_sr * self.duration
        if signal.shape[1] > length:
            signal = signal[:, :length]
        else:
            signal = torch.nn.functional.pad(signal, (0, length - signal.shape[1]))
        return signal

    def _resample(self, signal, sr):
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            signal = resampler(signal)
        return signal
    
    def _mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _get_audio_path(self, index: int) -> str:
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = f"data/UrbanSound8K/{fold}/{self.annotations.iloc[index, 0]}"
        return path
    
    def _get_label(self, index: int):
        return self.annotations.iloc[index, 6]
