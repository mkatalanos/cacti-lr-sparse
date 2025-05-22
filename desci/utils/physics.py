from typing import Callable, Tuple

import numpy as np
import scipy.io as scio


def generate_phi() -> Tuple[Callable, Callable]:
    """
    Generates two functions that take as argument a vector and return
    Φχ
    χ'Φ
    """
    def phix(x): return x
    def phity(y): return y
    return phix, phity
