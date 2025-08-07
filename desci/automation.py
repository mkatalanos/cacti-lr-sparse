import argparse

import pandas as pd
from lr_sparse_admm import ADMM
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from utils.dataloader import load_mat, load_video
import time
from utils.physics import generate_mask, phi
from utils.visualize import write_cube


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("lambda_0", type=float)
    parser.add_argument("lambda_1", type=float)
    parser.add_argument("lambda_2", type=float)
    parser.add_argument("lambda_3", type=float)
    parser.add_argument("-r", "--rho", type=float, default=1)
    parser.add_argument("-f", "--frames", type=int, default=8)
    parser.add_argument("-b", "--block", type=float, default=0.5)
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

    START_FRAME = 30
    name = "CASIA"
    x = load_video(
        "./datasets/video/casia_angleview_p01_jump_a1.mp4")[
        START_FRAME:START_FRAME+frames,
        :, :
    ]
    mask = generate_mask(x.shape, block)
    y = phi(x, mask)

    # x, mask, y = load_mat("./datasets/drop40_cacti.mat")
    # name = "DROP"
    # block = 0.5

    F, M, N = mask.shape

    print(
        f"Running with:, {lambda_0=}, {lambda_1=}, {
            lambda_2=}, {lambda_3=}, {frames=}, {rho=}, {block=}, {MAX_IT=}"
    )

    start_time = time.time()
    X, S, L, U, V, B, crits = ADMM(
        y,
        mask,
        rho=rho,
        lambda_0=lambda_0,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        MAX_IT=MAX_IT,
        verbose=False
    )
    end_time = time.time()

    PSNR = peak_signal_noise_ratio(x, X, data_range=255)
    SSIM = structural_similarity(x, X, data_range=255)
    duration = end_time-start_time

    print(f"{PSNR=:.2f},{SSIM=:.2f},{duration=:.2f}")

    out_title = f"out/{name}_F_{frames}_b{block: .2f}_l0_{lambda_0: .2f}_l1_{lambda_1: .2f}_l2_{
        lambda_2: .2f}_l3_{lambda_3: .2f}_r_{rho: .2f}_it_{MAX_IT}"

    columns = ["|Y-H(X)|", "|U|_1", "|L|_*", "|V|_*",
               "|S-U|", "|S-V|", "|X-B-V|", "primal", "dual"]
    df = pd.DataFrame(crits, columns=columns)
    df["PSNR"] = PSNR
    df["SSIM"] = SSIM
    df["duration"] = duration

    df.to_csv(f"{out_title}.csv")

    write_cube(X, f"{out_title}.png")
