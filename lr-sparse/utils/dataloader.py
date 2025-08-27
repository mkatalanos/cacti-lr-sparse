import imageio.v3 as iio
from scipy import io
import numpy as np


def load_video(fpath: str):
    video = iio.imread(fpath)
    grayscale_video = (
        0.2989 * video[:, :, :, 0]
        + 0.5870 * video[:, :, :, 1]
        + 0.1140 * video[:, :, :, 2]
    )

    # grayscale_video = np.transpose(grayscale_video, (1, 2, 0))

    return grayscale_video.round()


def load_mat(fpath: str):
    dataset = io.loadmat(fpath)
    x = dataset["orig"]
    mask = dataset["mask"]
    meas = dataset["meas"]

    M, N, F = mask.shape

    # Truncate to F
    # x = x[:, :, :F]
    meas = meas[:, :, 0]

    # Transpose to F,M,N
    x = x.transpose(2, 0, 1)
    mask = mask.transpose(2, 0, 1)
    meas = meas

    # Cast all to float64
    x = x.astype(np.float64)
    mask = mask.astype(np.float64)
    meas = meas.astype(np.float64)

    return x, mask, meas


if __name__ == "__main__":
    from utils.visualize import *

    # video = load_video("./datasets/video/casia_angleview_p01_jump_a1.mp4")
    x, mask, meas = load_mat("./datasets/traffic48_cacti.mat")
