from torch import nn

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

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        out = self.softmax(x)
        return out
