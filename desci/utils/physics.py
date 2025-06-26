from typing import Callable, Tuple

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

# from numba import jit, njit
from numpy.typing import NDArray
from scipy import io


def generate_mask(shape, block_rate=0.5, seed=None):
    shape_flat = np.prod(shape)
    np.random.seed(seed)
    mask = np.random.randint(0, 1000, int(shape_flat))
    mask[mask < block_rate * 1000] = 0
    mask[mask != 0] = 1
    mask = mask.reshape(shape)
    return mask


def init(dataset: str, sparsity=0.5):
    # Generate Measurement operator

    # TODO: Waiting for Shubham to generate own masks
    # according to device physics

    # Load data from Matlab file
    dataset = io.loadmat(dataset)

    x = dataset["orig"]
    mask = np.random.randint(0, 1000, x.shape)
    mask[mask < sparsity * 1000] = 0
    mask[mask != 0] = 1
    # mask = dataset["mask"]
    # meas = dataset["meas"]
    x, y = apply_cacti_mask_single(x, mask)
    # assert np.all(
    #     np.isclose(0, y - meas[:, :, 0])
    # ), "Measured signal doesn't match dataset"

    return x, y, mask


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
    x = np.repeat(y[:, :, np.newaxis], T, axis=2)
    return x.reshape(-1)


def generate_phi(
    mask: NDArray[np.uint8],
) -> Tuple[Callable, Callable, NDArray[np.uint8]]:
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
        H, W, T = mask.shape
        y = y.reshape(H, W)
        x = np.repeat(y[:, :, np.newaxis], T, axis=2)
        return x.reshape(-1)

    PhiPhit = mask.sum(axis=2).reshape(-1)

    return phix, phity, PhiPhit


def apply_cacti_mask(
    x: NDArray[np.uint8], mask: NDArray[np.uint8]
) -> NDArray[np.uint16]:
    """
    Applies a proper CACTI mask in chunks
    """
    H, W, B = x.shape
    Hm, Wm, T = mask.shape

    assert H == Hm, "Height dimensions must match"
    assert W == Wm, "Width dimensions must match"

    B_T = B // T
    y = np.zeros((H, W, B_T), dtype=np.uint16)
    for i in range(B_T):
        y[:, :, i] = np.sum(np.split(x, B_T, axis=2)[i] * mask, axis=2)

    return y


def apply_cacti_mask_single(
    x: NDArray[np.uint8], mask: NDArray[np.uint8]
) -> Tuple[NDArray[np.uint8], NDArray[np.uint16]]:
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


def A(x: cp.Variable, mask: NDArray[np.uint8]) -> cp.Expression:
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

    mask_flat = mask.reshape(H * W, B)
    diag_matrices = [sp.diags(mask_flat[:, b], dtype=np.uint8)
                     for b in range(B)]

    Phi = sp.hstack(diag_matrices, format="coo", dtype=np.uint8)

    return Phi
