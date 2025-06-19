import imageio.v3 as iio
import numpy as np


def load_video(fpath: str):
    video = iio.imread(fpath)
    grayscale_video = (
        0.2989 * video[:, :, :, 0]
        + 0.5870 * video[:, :, :, 1]
        + 0.1140 * video[:, :, :, 2]
    )

    grayscale_video = np.transpose(grayscale_video, (1, 2, 0)).round()

    return grayscale_video


if __name__ == "__main__":
    from utils.visualize import *

    video = load_video("./datasets/video/casia_angleview_p01_jump_a1.mp4")
