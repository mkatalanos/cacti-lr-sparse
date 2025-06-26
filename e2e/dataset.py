import imageio.v3 as iio
from typing import List
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset

from utils.physics import generate_mask, phi


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
    stride = B
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

    def __len__(self):
        return len(self.collected_slices)

    def __getitem__(self, idx):
        x = self.collected_slices[idx]

        # Generate random mask with idx as seed
        mask = generate_mask(x.shape, 0.2, idx)

        y = phi(x, mask)


if __name__ == "__main__":
    from utils.visualize import *

    video = load_frames(
        "./dataset/casia_angleview_p01_run_a1.mp4")
