import imageio.v3 as iio
from typing import List
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset

from utils.physics import generate_mask, phi, phit


def load_frames(fpath: str) -> NDArray:
    video = iio.imread(fpath)
    grayscale_video = (
        0.2989 * video[:, :, :, 0]
        + 0.5870 * video[:, :, :, 1]
        + 0.1140 * video[:, :, :, 2]
    ).round()

    return grayscale_video


def slice_video(video: NDArray, B=20):
    F, M, N = video.shape
    stride = B//4
    subvids = []
    for i in range(0, F-B+1, stride):
        subvids.append(video[i:i+B])
    return subvids


class VideoDataset(Dataset):
    def __init__(self, sources: List[str], B=20, block_rate=0.2):
        super().__init__()
        self.block_rate = block_rate
        videos = [load_frames(fpath) for fpath in sources]
        self.collected_slices = []

        for video in videos:
            self.collected_slices += slice_video(video, B)

    @staticmethod
    def normalize(x):
        return (x / 127.5) - 1

    @staticmethod
    def denormalize(x):
        return (x + 1) * 127.5

    def __len__(self):
        return len(self.collected_slices)

    def __getitem__(self, idx):
        x = self.collected_slices[idx].transpose(1, 2, 0)

        M, N, F = x.shape

        # Generate random mask with idx as seed
        mask = generate_mask(x.shape, self.block_rate, idx)

        y = phi(x, mask).reshape(M, N)

        mff = np.multiply(mask, mask, dtype=np.float64).sum(axis=2)
        mff[mff == 0] = 1e-8

        phiphit_inv = np.divide(y, mff)

        inverted = np.multiply(mask, phiphit_inv[:, :, np.newaxis])

        x = x.transpose(2, 0, 1)
        inverted = inverted.transpose(2, 0, 1)

        # Normalizing
        x = VideoDataset.normalize(x)
        inverted = VideoDataset.normalize(inverted)

        sample = {"truth": x, "inverted": inverted}
        return sample


if __name__ == "__main__":
    from utils.visualize import *

    video = load_frames(
        "./dataset/casia_angleview_p01_run_a1.mp4")
    collected_slices = slice_video(video)
    idx = 2
    x = collected_slices[idx].transpose(1, 2, 0)
