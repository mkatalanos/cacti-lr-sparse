import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# lr_sparse_admm.py
from lr_sparse_admm import update_S


@pytest.fixture
def sample_data():
    # Randomly generated data
    M, N, F = 32, 32, 8  # Frame size and number of frames
    x = np.random.randint(0, 256, (M, N, F), dtype=np.uint8)
    mask = np.random.randint(0, 256, (M, N, F), dtype=np.uint8)
    y = np.multiply(mask, x).sum(axis=2)

    return x, y, mask, (M, N, F)


# Testing that S can run
def test_S_update_run(sample_data):
    x, y, mask, (M, N, F) = sample_data

    Y = y
    B = np.random.randint(0, 256, mask.shape)
    mask = mask
    U = B.copy()
    V = B.copy()
    Theta = B.copy()
    rho = 0.3

    _ = update_S(Y, B, mask, U, V, Theta, rho)

    assert True, "Code couldn't run"
