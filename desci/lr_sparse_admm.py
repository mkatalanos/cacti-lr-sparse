from typing import Any, Callable, Tuple

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_svd

from main import init
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

    Y_b = Y-phi(B).reshape((M, N))
    C1 = rho*(U+V-1/rho * Theta)
    C2 = np.zeros((M, N, F))
    for f in range(F):
        C2[:, :, f] = np.multiply(mask[:, :, f], Y_b)

    C3 = C1 + C2
    S = np.zeros((M, N, F))

    mff = np.multiply(mask, mask)

    for i in range(F):
        S[:, :, f] = np.multiply(np.divide(1, mff[:, :, f]+2*rho), C3[:, :, f])

    return S


def update_L(B, Delta, rho, lambda_2, mask, delta=1e-3,
             epsilon=1e-3, max_it=1000, svd_l=10, ):
    M, N, F = mask.shape
    La = B + (1/rho)*Delta
    La_bar = bar(La)
    u, s, vh = randomized_svd(La_bar, svd_l)
    # s is array of components
    dold = s.copy()

    for t in range(max_it):
        d = np.max(s-(lambda_2/rho)*(1/(dold+epsilon)), 0)
        if np.max(np.linalg.norm(d-dold, axis=1)) <= delta:
            break

    L_bar = u @ np.diag(d) @ vh

    L = L_bar.reshape(M, N, F)
    return L


def update_U():
    pass


def update_V():
    pass


def update_B():
    pass


if __name__ == "__main__":
    x, y, mask = init(dataset="./datasets/traffic48_cacti.mat")
    M, N, F = mask.shape
    phi, phit, phiphit = generate_phi(mask)
    # x = x.reshape(-1)
    # y = y.reshape(-1)
    # Testing S run

    Y = y
    B = np.random.randint(0, 256, mask.shape)
    mask = mask
    U = B.copy()
    V = B.copy()
    Theta = B.copy()
    rho = 0.3

    S = update_S(Y, B, mask, U, V, Theta, rho)

    raise SystemExit
