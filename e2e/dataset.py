import imageio.v3 as iio
import imageio_ffmpeg as iff
from typing import List
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset
from itertools import chain

from utils.physics import generate_mask, phi, phit


def load_frames(fpath: str) -> NDArray:
    video = iio.imread(fpath)
    grayscale_video = (
        0.2989 * video[:, :, :, 0]
        + 0.5870 * video[:, :, :, 1]
        + 0.1140 * video[:, :, :, 2]
    ).round()

    return torch.from_numpy(grayscale_video)


def load_nframes(fpath: str) -> int:
    nframes, secs = iff.count_frames_and_secs(fpath)
    return nframes


def dry_slice(nframes: int, B=20):
    stride = B
    slices = []
    for i in range(0, nframes-B+1, stride):
        slices.append((i, i+B))
    return slices


def dry_slice_video(fpath: str, B=20):
    nframes = load_nframes(fpath)
    slices = dry_slice(nframes, B)
    return [(fpath, subvid) for subvid in slices]


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
        self.slices = list(chain.from_iterable(
            [dry_slice_video(fpath, B) for fpath in sources]))

    @staticmethod
    def normalize(x):
        return (x / 127.5) - 1

    @staticmethod
    def denormalize(x):
        return (x + 1) * 127.5

    def load_slice_pair(self, idx):
        fpath, (i, j) = self.slices[idx]
        video = load_frames(fpath)
        return video[i:j]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        x = self.load_slice_pair(idx)

        F, M, N = x.shape

        # Generate random mask with idx as seed
        mask = generate_mask(x.shape, self.block_rate, idx)
        mask = torch.from_numpy(mask)

        y = torch.multiply(mask, x).sum(axis=0)

        mff = torch.multiply(mask, mask).sum(axis=0)
        # mff = np.multiply(mask, mask, dtype=np.float64).sum(axis=2)
        mff[mff == 0] = 1e-8

        phiphit_inv = torch.divide(y, mff)

        inverted = torch.multiply(mask, phiphit_inv[torch.newaxis, :, :])

        # x = x.transpose(2, 0, 1)
        # inverted = inverted.transpose(2, 0, 1)

        # Normalizing
        x = VideoDataset.normalize(x).to(torch.float)
        inverted = VideoDataset.normalize(inverted).to(torch.float)

        sample = {"truth": x, "inverted": inverted}
        return sample


if __name__ == "__main__":
    from utils.visualize import *
    import glob

    video = load_frames(
        "./dataset/casia_angleview_p01_run_a1.mp4")
    collected_slices = slice_video(video)
    idx = 2
    sources = glob.glob("./dataset/*.mp4")
    dataset = VideoDataset(sources)
    # vid = sources[0]
