import yaml
import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, MFCC, Spectrogram
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from dvclive import Live
from dvclive.lightning import DVCLiveLogger

from urbansound_classifier.dataset import AudioDataset
from urbansound_classifier.model import LitCNN

def train():
    params = yaml.safe_load(open("params.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = {
        "mel_spectrogram": MelSpectrogram,
        "mfcc": MFCC,
        "spectrogram": Spectrogram,
    }[params["transform"]["type"]]

    transform_params = {k:v for k, v in params["transform"]["params"].items() if k in transform.__init__.__code__.co_varnames}
    
    transform = transform(**transform_params).to(device)

    train_dataset = AudioDataset(transform, val=False, device=device)
    val_dataset = AudioDataset(transform, val=True, device=device)

    train_dataloader = DataLoader(train_dataset, batch_size=params["train"]["batch_size"], shuffle=params["train"]["shuffle"])
    val_dataloader = DataLoader(val_dataset, batch_size=params["train"]["batch_size"], shuffle=False)

    with Live(save_dvc_exp=True) as live:
        checkpoint_callback = ModelCheckpoint(
            dirpath="model",
            filename="best",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
        )
        model = LitCNN()
        trainer = Trainer(
            logger=DVCLiveLogger(),
            max_epochs=params["train"]["epochs"],
            callbacks=[checkpoint_callback],
            log_every_n_steps=1,
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        live.log_artifact(
            checkpoint_callback.best_model_path,
            type="model",
            name="best",
        )

if __name__ == '__main__':
    train()
