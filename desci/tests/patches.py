import numpy as np
import pytest
from utils.patches import extract_patches, reconstruct_from_patches


@pytest.fixture
def sample_data():
    # Randomly generated data
    M, N, F = 32, 32, 8  # Frame size and number of frames
    x = np.random.randint(0, 256, (M, N, F), dtype=np.uint8)
    return x, (M, N, F)


def test_patch_extract(sample_data):
    x, (M, N, F) = sample_data

    # Extract patches
    patch_size = 4
    X_tilde = extract_patches(x, patch_size)
    assert X_tilde.shape == (patch_size*patch_size,
                             M*N*F/(patch_size*patch_size))

    # Extract patches
    patch_size = 2
    X_tilde = extract_patches(x, patch_size)
    assert X_tilde.shape == (patch_size*patch_size,
                             M*N*F/(patch_size*patch_size))


def test_patch_extract_reconstruct(sample_data):
    x, (M, N, F) = sample_data
    patch_size = 4

    # Extract patches
    X_tilde = extract_patches(x, patch_size)

    # Reconstruct
    X_rec = reconstruct_from_patches(X_tilde, patch_size, (M, N, F))

    np.testing.assert_array_equal(x, X_rec)
