import numpy as np
import pytest
from utils.physics import apply_cacti_mask, apply_cacti_mask_single


@pytest.fixture
def sample_data():
    # Randomly generated data
    H, W, B, T = 4, 4, 8, 2
    x = np.random.randint(0, 256, (H, W, B), dtype=np.uint8)
    mask = np.random.randint(0, 2, (H, W, T), dtype=np.uint8)
    return x, mask, T


def test_check_trunc_x(sample_data):
    x, mask, T = sample_data

    x_trunc, y_trunc = apply_cacti_mask_single(x, mask)

    np.testing.assert_array_equal(x_trunc, x[:, :, 0:T])


def test_check_trunc_y(sample_data):
    x, mask, T = sample_data

    y_full = apply_cacti_mask(x, mask)

    x_trunc, y_trunc = apply_cacti_mask_single(x, mask)

    print(f"{y_full.shape=}, {y_trunc.shape=}, {y_full[:, :, 0].shape=}")
    np.testing.assert_array_equal(y_trunc, y_full[:, :, 0])
