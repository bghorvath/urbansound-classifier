import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import yaml

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.dense_layers(x)
        out = self.softmax(logits)
        return out

def train():
    params = yaml.safe_load(open("params.yaml"))

    train_data = MNIST(root="data", train=True, download=True, transform=ToTensor())
    dataloader = DataLoader(train_data, batch_size=params["train"]["batch_size"], shuffle=params["train"]["shuffle"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeedForward().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["train"]["lr"])
    loss_fn = nn.CrossEntropyLoss()

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