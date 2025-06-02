import cvxpy as cp
import numpy as np
import scipy.io as io
import scipy.sparse as sp
from numpy.typing import NDArray
from utils.physics import (apply_cacti_mask, apply_cacti_mask_single,
                           phi_from_mask)
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
    x, y = apply_cacti_mask_single(x, mask)
    assert np.all(np.isclose(0, y-meas[:, :, 0])
                  ), "Measured signal doesn't match dataset"

    return x, y, mask


def cvxpy_problem(x: NDArray[np.uint8], y: NDArray[np.uint16],
                  mask: NDArray[np.uint8]):
    """
    Define and solve a convexpy problem based on NNM
    |AX-y|_2^2 + lambda |X|_*
    """

    H, W, B = x.shape
    Hm, Wm, T = mask.shape

    assert H == Hm, "Height dimensions must match"
    assert W == Wm, "Width dimensions must match"

    var_x = cp.Variable((H, W, B))

    def A(x: cp.Variable):
        """
        Forward Operator rewritten to work with cvxpy
        """
        y = cp.multiply(x, mask).sum(axis=2)
        return y

    lambda_param = 1.0

    data_fidelity = cp.sum_squares(y-A(var_x))
    tv_reg = cp.sum([cp.tv(var_x[:, :, i]) for i in range(B)])

    objective = cp.Minimize(0.5 * data_fidelity + lambda_param * tv_reg)
    problem = cp.Problem(objective)

    return problem


if __name__ == "__main__":
    x, y, mask = init(dataset="./datasets/traffic48_cacti.mat")
    problem = cvxpy_problem(x, y, mask)
    problem.solve(solver="SCS")
    raise SystemExit
