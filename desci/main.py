import numpy as np
import scipy.io as io

import utils.physics
from utils.visualize import visualize_cube


def init():
    # Generate Measurement operator
    Phix, Phity = utils.physics.generate_phi()

    # Load data from Matlab file
    dataset = io.loadmat("./datasets/kobe32_cacti.mat")

    x = dataset['orig']

    return x, Phix, Phity


def main():
    Phix, Phity = utils.physics.generate_phi()


if __name__ == "__main__":
    x, _, _ = init()
    visualize_cube(x)
    raise SystemExit
