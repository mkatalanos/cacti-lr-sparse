import argparse

from lr_sparse_admm import ADMM

from utils.dataloader import load_video
from utils.physics import generate_mask, phi
from utils.visualize import write_cube

import pandas as pd
from skimage.metrics import peak_signal_noise_ratio


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("lambda_0", type=float)
    parser.add_argument("lambda_1", type=float)
    parser.add_argument("lambda_2", type=float)
    parser.add_argument("lambda_3", type=float)
    parser.add_argument("-r", "--rho", type=float, default=1)
    parser.add_argument("-f", "--frames", type=int, default=8)
    parser.add_argument("-b", "--block", type=float, default=0.2)
    parser.add_argument("-i", "--iterations", type=int, default=500)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = process_args()

    lambda_0 = args.lambda_0
    lambda_1 = args.lambda_1
    lambda_2 = args.lambda_2
    lambda_3 = args.lambda_3

    frames = args.frames
    block = args.block
    rho = args.rho
    MAX_IT = args.iterations

    assert block < 1

    x = load_video("./datasets/video/casia_angleview_p01_jump_a1.mp4")[
        :, :, 30: (30 + frames)
    ]

    mask = generate_mask(x.shape, block)
    y = phi(x, mask)

    M, N, F = mask.shape

    lambda_0 /= M * N
    lambda_1 /= M * N * F
    lambda_2 /= 10
    lambda_3 /= 20

    print(
        f"Running with:, {lambda_0=}, {lambda_1=}, {
            lambda_2=}, {lambda_3=}, {frames=}, {rho=}, {block=}, {MAX_IT=}"
    )

    X, S, L, U, V, B, crits = ADMM(
        y,
        mask,
        rho=rho,
        lambda_0=lambda_0,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        MAX_IT=MAX_IT,
    )
    psnr = peak_signal_noise_ratio(x, X, data_range=255)

    out_title = f"out/l0_{lambda_0:.2e}_l1_{lambda_1:.2e}_l2_{
        lambda_2:.2e}_l3_{lambda_3:.2e}_r_{rho:.2e}_it_{MAX_IT}"

    columns = ["|Y-H(X)|", "|U|_1", "|L|_*", "|V|_*",
               "|S-U|", "|S-V|", "|X-B-V|"]
    df = pd.DataFrame(crits, columns=columns)
    df["PSNR"] = psnr

    df.to_csv(f"{out_title}.csv")

    write_cube(X, f"{out_title}.png")
