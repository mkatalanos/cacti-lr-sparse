from typing import Any, Callable, Tuple

# External libraries
import cvxpy as cp
import numpy as np
from main import init
from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_svd
from utils.patches import extract_patches, find_similar
from utils.physics import generate_phi
from utils.visualize import visualize_cube


def desci(
        y: NDArray[np.uint16],
        mask: NDArray[np.uint8],
        MAX_IT: int = 3,
        PATCH_SIZE: int = 16,
        M: int = 20,
):
    H, W, B = mask.shape

    Phi, Phit, PhiPhit = generate_phi(mask)

    lambda_var = 0.5
    gamma_var = 0.5
    c_var = 0.5
    epsilon = 1e-16
    beta_var = 0

    x = Phit(y)

    for it in range(MAX_IT):
        theta_var = theta(y, x, beta_var, gamma_var, PhiPhit, Phi, Phit)
        q_var = theta_var - beta_var

        # Patch stuff
        patches = extract_patches(q_var.reshape(H, W, B), PATCH_SIZE)

        for patch in patches:
            R = find_similar(patch, patches, M)
            U, s, Vh = randomized_svd(R, 8)
        beta_var = beta_var-(theta_var-x)

    return x


def theta(y: NDArray,
          x_t: NDArray,
          b_t: float,
          gamma: float,
          phiphit: NDArray,
          phi_x: Callable[NDArray, NDArray],
          phit_y: Callable[NDArray, NDArray]):
    theta = (x_t+b_t).view(np.float64)

    phi_xb: NDArray = phi_x(theta)

    frac = np.divide(y-phi_xb, gamma+phiphit)

    theta += phit_y(frac)

    return theta


if __name__ == "__main__":
    x, y, mask = init(dataset="./datasets/traffic48_cacti.mat")
    phix, phity, phiphit = generate_phi(mask)
    x = x.reshape(-1)
    y, yold = phix(x), y.reshape(-1)

    np.testing.assert_allclose(y, yold.reshape(-1))

    raise SystemExit
