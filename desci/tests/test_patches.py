import numpy as np
import pytest
from lr_sparse_admm import soft_thresh
from utils.patches import (
    extract_patches,
    extract_sparse_patches,
    reconstruct_from_patches,
    reconstruct_sparse_patches,
)


@pytest.fixture
def sample_data():
    # Randomly generated data
    M, N, F = 256, 256, 8  # Frame size and number of frames
    x = np.random.randint(0, 256, (F, M, N), dtype=np.uint8)
    return x, (F, M, N)


def test_patch_extract_old(sample_data):
    x, (F, M, N) = sample_data

    # Extract patches
    patch_size = 4
    X_tilde = extract_patches(x, patch_size)
    assert X_tilde.shape == (
        patch_size * patch_size,
        M * N * F / (patch_size * patch_size),
    )

    # Extract patches
    patch_size = 2
    X_tilde = extract_patches(x, patch_size)
    assert X_tilde.shape == (
        patch_size * patch_size,
        M * N * F / (patch_size * patch_size),
    )


def test_patch_extract_reconstruct_old(sample_data):
    x, (F, M, N) = sample_data
    patch_size = 4

    # Extract patches
    X_tilde = extract_patches(x, patch_size)

    # Reconstruct
    X_rec = reconstruct_from_patches(X_tilde, patch_size, (F, M, N))

    np.testing.assert_array_equal(x, X_rec)


def test_patch_extract(sample_data):
    x, (F, M, N) = sample_data

    # Extract patches
    patch_size = 4
    X_tilde, locations = extract_sparse_patches(x, patch_size)
    assert X_tilde.shape == (
        patch_size * patch_size,
        M * N * F / (patch_size * patch_size),
    )

    # Extract patches
    patch_size = 2
    X_tilde, locations = extract_sparse_patches(x, patch_size)
    assert X_tilde.shape == (
        patch_size * patch_size,
        M * N * F / (patch_size * patch_size),
    )


def test_patch_extract_reconstruct(sample_data):
    x, (F, M, N) = sample_data
    patch_size = 4

    # Extract patches
    X_tilde, locations = extract_sparse_patches(x, patch_size)

    # Reconstruct
    X_rec = reconstruct_sparse_patches(X_tilde, locations, (F, M, N))

    np.testing.assert_array_equal(x, X_rec)


def test_patch_extract_sparse(sample_data):
    x, (F, M, N) = sample_data

    x = soft_thresh(x, 240)

    # Extract patches
    patch_size = 4
    X_tilde, locations = extract_sparse_patches(x, patch_size)
    assert X_tilde.shape[1] < M * N * F / (patch_size * patch_size)

    # Extract patches
    patch_size = 2
    X_tilde, locations = extract_sparse_patches(x, patch_size)
    assert X_tilde.shape[1] < M * N * F / (patch_size * patch_size)


def test_patch_extract_reconstruct_sparse(sample_data):
    x, (F, M, N) = sample_data
    patch_size = 4
    x = soft_thresh(x, 220)

    # Extract patches
    X_tilde, locations = extract_sparse_patches(x, patch_size)

    # Reconstruct
    X_rec = reconstruct_sparse_patches(X_tilde, locations, (F, M, N))

    np.testing.assert_array_equal(x, X_rec)
