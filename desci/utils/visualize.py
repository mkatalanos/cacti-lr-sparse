import numpy as np
from PIL import Image


def visualize_cube(img):
    if len(img.shape) != 3:
        img = img[:, np.newaxis]
    assert len(img.shape) == 3

    H, W, F = img.shape
    side_by_side = np.zeros((H, W * F), dtype=np.uint8)

    for i in range(F):
        side_by_side[:, i*W:(i+1)*W] = img[:, :, i]

    Image.fromarray(side_by_side).show()
