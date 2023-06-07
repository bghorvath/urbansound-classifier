import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader

from torchaudio.transforms import MelSpectrogram, MFCC, Spectrogram
from torchaudio_demo.dataset import AudioDataset
from torchaudio_demo.model import CNN

def train():
    params = yaml.safe_load(open("params.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = {
        "mel_spectrogram": MelSpectrogram,
        "mfcc": MFCC,
        "spectrogram": Spectrogram,
    }[params["transform"]["type"]]

    transform_params = {k:v for k, v in params["transform"]["params"].items() if k in transform.__init__.__code__.co_varnames}
    
    transform = transform(**transform_params)

    dataset = AudioDataset(transform, device=device)

    dataloader = DataLoader(dataset, batch_size=params["train"]["batch_size"], shuffle=params["train"]["shuffle"])
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["train"]["lr"])
    loss_fn = nn.CrossEntropyLoss()
    
    loss = torch.tensor(0.0)
    for epoch in range(params["train"]["epochs"]):
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            x_hat = model(x)
            loss = loss_fn(x_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    print("Done!")

    torch.save(model.state_dict(), "model.pth")

if __name__ == '__main__':
    train()
