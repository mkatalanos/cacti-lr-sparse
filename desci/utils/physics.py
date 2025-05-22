from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray


def generate_phi() -> Tuple[Callable, Callable]:
    """
    Generates two functions that take as argument a vector and return
    Φχ
    Φ'y
    """
    def phix(x): return x
    def phity(y): return y
    return phix, phity


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
        y[:, :, i] = np.sum(np.dsplit(x, B_T)[i]*mask, axis=2)

    return y
