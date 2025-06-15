import cvxpy as cp
import numpy as np
import scipy.io as io
import scipy.sparse as sp
from numpy.typing import NDArray
from utils.patches import extract_patches, find_similar
from utils.physics import apply_cacti_mask, apply_cacti_mask_single, phi_from_mask
from utils.visualize import visualize_cube


def init(dataset: str):
    # Generate Measurement operator

    # TODO: Waiting for Shubham to generate own masks
    # according to device physics

    # Load data from Matlab file
    dataset = io.loadmat(dataset)

    x = dataset["orig"]
    mask = dataset["mask"]
    meas = dataset["meas"]
    x, y = apply_cacti_mask_single(x, mask)
    assert np.all(
        np.isclose(0, y - meas[:, :, 0])
    ), "Measured signal doesn't match dataset"

    return x, y, mask


def cvxpy_split(x: NDArray[np.uint8]):
    """ """

    H, W, B = x.shape
    x_bar = x.reshape(H * W, B)

    var_b = cp.Variable((H * W, B))
    var_s = cp.Variable((H * W, B))

    lambda_param = 0.1

    r1 = cp.norm(var_b, "nuc")
    r2 = cp.norm(var_s.flatten("C"), 1)

    objective = cp.Minimize(r1 + lambda_param * r2)
    constraints = [x_bar == var_b + var_s]
    problem = cp.Problem(objective, constraints)

    return problem


def cvxpy_tv_problem(
    x: NDArray[np.uint8], y: NDArray[np.uint16], mask: NDArray[np.uint8]
):
    """
    Define and solve a convexpy problem based on TV Norm
    |AX-y|_2^2 + lambda |X|_TV
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

    data_fidelity = cp.sum_squares(y - A(var_x))
    tv_reg = cp.sum([cp.tv(var_x[:, :, i]) for i in range(B)])

    objective = cp.Minimize(0.5 * data_fidelity + lambda_param * tv_reg)
    problem = cp.Problem(objective)

    return problem


if __name__ == "__main__":
    x, y, mask = init(dataset="./datasets/traffic48_cacti.mat")
    H, W, B = mask.shape
    x = x[:32, :32, :]

    problem = cvxpy_split(x)
    problem.solve("SCS")
    b = problem.variables()[0].value.reshape(32, 32, 8)
    s = problem.variables()[1].value.reshape(32, 32, 8)
    visualize_cube(b)
    visualize_cube(s)
    # patch_size = 64
    # patches = extract_patches(x, patch_size)
    # print(f"Number of patches {patches.shape[1]}")
    # sim_patches = find_similar(patches[:, 25], patches, 8)
    #
    # visualize_cube(sim_patches.reshape(patch_size, patch_size, -1))

    raise SystemExit
