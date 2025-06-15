import numpy as np
import pytest
from lr_sparse_admm import update_B, update_L, update_S, update_U, update_V


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
    Gamma = B.copy()
    rho = 0.3

    S = update_S(Y, B, mask, U, V, Theta, Gamma, rho)
    assert S.shape == (M, N, F), "Shape of S must be MxNxF"


def test_L_update_run(sample_data):
    x, y, mask, (M, N, F) = sample_data

    B = np.random.randint(0, 256, mask.shape)
    Delta = np.random.randint(0, 256, mask.shape)
    rho = 0.3
    lambda_2 = 0.5
    L = update_L(B, Delta, rho, lambda_2, mask, max_it=3)

    assert L.shape == (M, N, F), "Shape of L must be MxNxF"


def test_U_update_run(sample_data):
    x, y, mask, (M, N, F) = sample_data

    S = np.random.randint(0, 256, mask.shape)
    Theta = np.random.randint(0, 256, mask.shape)
    rho = 0.3
    lambda_1 = 0.5
    U = update_U(S, Theta, lambda_1, rho)

    assert U.shape == (M, N, F), "Shape of U must be MxNxF"


def test_B_update_run(sample_data):
    x, y, mask, (M, N, F) = sample_data

    # Testing B run
    Y = y
    mask = mask
    S = np.random.randint(0, 256, mask.shape)
    L = S.copy()
    Delta = S.copy()
    rho = 0.3

    B = update_B(Y, mask, S, L, Delta, rho)

    assert B.shape == (M, N, F), "Shape of B must be MxNxF"


def test_V_update_run(sample_data):
    x, y, mask, (M, N, F) = sample_data

    S = np.random.randint(0, 256, mask.shape)
    Gamma = np.random.randint(0, 256, mask.shape)
    rho = 0.3
    lambda_3 = 0.5
    V = update_V(S, Gamma, rho, lambda_3, 3)

    assert V.shape == (M, N, F), "Shape of V must be MxNxF"
