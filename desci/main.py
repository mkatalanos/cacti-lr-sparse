
import numpy as np
import scipy.io as io
import scipy.sparse as sp

from utils.physics import apply_cacti_mask, phi_from_mask
from utils.visualize import visualize_cube


def init(dataset: str):
    # Generate Measurement operator

    # TODO: Waiting for Shubham to generate own masks
    # according to device physics

    # Load data from Matlab file
    dataset = io.loadmat(dataset)

    x = dataset['orig']
    mask = dataset['mask']
    meas = dataset['meas']
    y = apply_cacti_mask(x, mask)
    assert np.all(np.isclose(0, y-meas)
                  ), "Measured signal doesn't match dataset"

    return x, y, mask


def gap_solve(y, mask):
    pass


if __name__ == "__main__":
    x, y, mask = init(dataset="./datasets/traffic48_cacti.mat")

    # visualize_cube(x)
    Phi = phi_from_mask(mask)
    Phit = Phi.T
    PhiPhit = (Phi @ Phit)

    y2 = Phi @ x
    raise SystemExit
