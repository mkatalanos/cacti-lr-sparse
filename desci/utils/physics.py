from typing import Callable, Tuple

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
# from numba import jit, njit
from numpy.typing import NDArray


# @njit
def phi(x: NDArray, mask: NDArray):
    """
    Applies forward transform
    x,mask
    """
    H, W, T = mask.shape
    # x = x.reshape(H, W, -1)
    y = np.multiply(mask, x).sum(axis=2).reshape(-1)
    return y


def phit(y, mask):
    """
    Applies the adjoint of the transform
    y,mask
    """
    H, W, T = mask.shape
    y = y.reshape(H, W)
    x = np.divide(y[:, :, None], mask)
    x[np.isnan(x)] = 0
    x[np.isinf(x)] = 0
    x /= x.max()
    x *= 255
    return x.reshape(-1)


def generate_phi(mask: NDArray[np.uint8]) -> Tuple[Callable, Callable, NDArray[np.uint8]]:
    """
    Generates two functions that take as argument a vector and return
    Φχ
    Φ'y
    ΦΦ'
    """
    H, W, T = mask.shape

    def phix(x):
        return np.multiply(mask, x.reshape(H, W, -1)).sum(axis=2).reshape(-1)

    def phity(y):
        y = y.reshape(H, W)
        x = mask * y[:, :, None]
        return x.reshape(-1)

    # H, W, B = mask.shape
    # flat_mask = mask.reshape(-1, B)
    #
    # PhiPhit = np.dot(flat_mask, flat_mask.T)

    # PhiPhit = (mask**2).sum(axis=2).reshape(-1)
    PhiPhit = mask.sum(axis=2).reshape(-1)

    return phix, phity, PhiPhit


def apply_cacti_mask(x: NDArray[np.uint8], mask: NDArray[np.uint8]) \
        -> NDArray[np.uint16]:
    """
    Applies a proper CACTI mask in chunks
    """
    H, W, B = x.shape
    Hm, Wm, T = mask.shape

    assert H == Hm, "Height dimensions must match"
    assert W == Wm, "Width dimensions must match"

    B_T = B//T
    y = np.zeros((H, W, B_T), dtype=np.uint16)
    for i in range(B_T):
        y[:, :, i] = np.sum(np.split(x, B_T, axis=2)[i]*mask, axis=2)

    return y


def apply_cacti_mask_single(x: NDArray[np.uint8], mask: NDArray[np.uint8]) \
        -> Tuple[NDArray[np.uint8], NDArray[np.uint16]]:
    """
    Applies a cacti mask to the first mask length chunk.
    """
    H, W, B = x.shape
    Hm, Wm, T = mask.shape

    assert H == Hm, "Height dimensions must match"
    assert W == Wm, "Width dimensions must match"

    x_trunc = x[:, :, :T]
    y = np.multiply(x_trunc, mask).sum(axis=2, dtype=np.uint16)

    return x_trunc, y


def A(x: cp.Variable, mask: NDArray[np.uint8]) \
        -> cp.Expression:
    """
    Applies a cacti mask to the first mask length chunk.
    """
    H, W, B = x.shape
    Hm, Wm, T = mask.shape

    assert H == Hm, "Height dimensions must match"
    assert W == Wm, "Width dimensions must match"

    x_trunc = x[:, :, :T]
    y = cp.multiply(x_trunc, mask).sum(axis=2)

    return y


def phi_from_mask(mask: NDArray[np.uint8]):
    H, W, B = mask.shape

    mask_flat = mask.reshape(H*W, B)
    diag_matrices = [sp.diags(mask_flat[:, b], dtype=np.uint8)
                     for b in range(B)]

    Phi = sp.hstack(diag_matrices, format="coo", dtype=np.uint8)

    return Phi
