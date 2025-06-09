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


def bar(x: NDArray) -> NDArray:
    assert len(x.shape) == 3
    M, N, F = x.shape

    x_bar = np.zeros((M*N, F))

    for f in range(F):
        x_bar[:, f] = x[:, :, f].reshape(-1)

    return x_bar


def tilde(x: NDArray, patch_size: int = 16) -> NDArray:
    assert len(x.shape) == 3
    return extract_patches(x, patch_size)


def update_S(Y, B, mask, U, V, Theta, rho):
    phi, phit, phiphit = generate_phi(mask)
    M, N, F = mask.shape

    Y_b = Y-phi(B)
    C1 = rho*(U+V-1/rho * Theta)
    C2 = np.zeros((M, N, F))

    pass


if __name__ == "__main__":
    x, y, mask = init(dataset="./datasets/traffic48_cacti.mat")
    phi, phit, phiphit = generate_phi(mask)
    x = x.reshape(-1)
    y = y.reshape(-1)

    raise SystemExit
