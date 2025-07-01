import pytorch_lightning as pl
from skimage.metrics import peak_signal_noise_ratio
import torch

from model import E2E_CNN
from loss import CustomLoss

import pickle


class CustomModel(pl.LightningModule):
    def __init__(self, B=20, channels=64, depth=5):
        super().__init__()

        self.layer = E2E_CNN(B, channels, depth)

        self.lr = 1e-3
        self.loss = CustomLoss(alpha=1, beta=0.1, B=B)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        xs = batch['truth']
        ys = batch['inverted']

        pred = self(ys)

        loss = self.loss(pred, xs)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        xs = batch['truth']
        ys = batch['inverted']

        pred = self(ys)

        loss = self.loss(pred, xs)

        # Scale both to 0-255
        xs = (xs + 1) * 127.5
        pred = (pred + 1) * 127.5

        # Move to cpu and round to closest int
        xs = xs.cpu().numpy().round()
        pred = pred.cpu().numpy().round()

        psnr = peak_signal_noise_ratio(pred, xs, data_range=255)

        self.log("val_loss", loss)
        self.log("val_psnr", psnr)

    def test_step(self, batch, batch_idx):
        xs = batch['truth']
        ys = batch['inverted']

        pred = self(ys)

        loss = self.loss(pred, xs)

        # Scale both to 0-255
        xs = (xs + 1) * 127.5
        pred = (pred + 1) * 127.5

        # Move to cpu and round to closest int
        xs = xs.cpu().numpy().round()
        pred = pred.cpu().numpy().round()

        psnr = peak_signal_noise_ratio(pred, xs, data_range=255)

        self.log("test_loss", loss)
        self.log("test_psnr", psnr)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt
