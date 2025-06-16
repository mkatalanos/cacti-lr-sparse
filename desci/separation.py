from rpca import RobustPCA
import numpy as np
from utils.physics import init, phi
from utils.dataloader import load_video
from utils.visualize import visualize_cube


if __name__ == "__main__":
    x = load_video("./datasets/video/casia_angleview_p01_jump_a1.mp4")

    M, N, F = x.shape

    rpca = RobustPCA(trim=True)

    x_bar = x.reshape(M * N, F)
    rpca.fit(x_bar)
    S = rpca.S_.reshape(M, N, F)
    L = rpca.L_.reshape(M, N, F)
