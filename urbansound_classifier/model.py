import torch
from torch import nn
import lightning.pytorch as pl

class LitFF(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(64 * 44, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.dense_layers(x)
        out = self.softmax(logits)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class LitCNN(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=0.0):
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
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        logits = self.linear(x)
        out = self.softmax(logits)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
