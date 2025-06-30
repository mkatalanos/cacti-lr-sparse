import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader
from typing import List
from dataset import VideoDataset
import glob


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 4, num_workers: int = 16,
                 B=20, block_rate=0.2):
        super().__init__()
        self.sources = glob.glob(f"{data_dir}/*.mp4")
        self.B = B
        self.block_rate = block_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        self.train_sources, self.val_sources, self.test_sources = random_split(
            self.sources, [0.6, 0.2, 0.2], torch.Generator().manual_seed(2025))

        if stage == "fit":
            self.train_dataset = VideoDataset(
                self.train_sources, self.B, self.block_rate)
            self.val_dataset = VideoDataset(
                self.val_sources, self.B, self.block_rate)
        elif stage == "validate":
            self.val_dataset = VideoDataset(
                self.val_sources, self.B, self.block_rate)
        elif stage == "test":
            self.test_dataset = VideoDataset(
                self.test_sources, self.B, self.block_rate)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    vdm = VideoDataModule("./dataset/")
