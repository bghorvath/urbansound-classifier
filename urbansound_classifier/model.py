from torch import nn
from torchinfo import summary

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
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2), # padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # stride
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128*5*4, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        logits = self.linear(x)
        out = self.softmax(logits)
        return out
    

if __name__ == "__main__":
    model = CNN()
    summary(model, input_size=(1, 1, 64, 44))
