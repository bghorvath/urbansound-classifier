import yaml
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC, Spectrogram

class AudioDataset(Dataset):
    def __init__(self, transform) -> None:
        self.annotations = pd.read_csv("data/UrbanSound8K/UrbanSound8K.csv")
        self.transform = transform
        self.target_sr = yaml.safe_load(open("params.yaml"))["transform"]["params"]["sample_rate"]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int):
        audio_path = self._get_audio_path(index)
        label = self._get_label(index)
        signal, sr = torchaudio.load(audio_path)
        signal = self._resample(signal, sr)
        signal = self._mix_down(signal)
        signal = self.transform(signal)
        return signal, label

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
    
    def _get_label(self, index: int) -> int:
        return self.annotations.iloc[index, 6]

if __name__ == '__main__':
    params = yaml.safe_load(open("params.yaml"))

    transform = {
        "mel_spectrogram": MelSpectrogram,
        "mfcc": MFCC,
        "spectrogram": Spectrogram,
    }[params["transform"]["type"]]

    transform_params = {k:v for k, v in params["transform"]["params"].items() if k in transform.__init__.__code__.co_varnames}
    
    transform = transform(**transform_params)

    dataset = AudioDataset(transform)
    print(dataset[0])
