import numpy as np
from PIL import Image


def visualize_cube(img):
    shape = img.shape
    side_by_side = np.zeros((shape[0], shape[1] * shape[2]), dtype=np.uint8)

    for i in range(shape[2]):
        side_by_side[:, i*shape[1]:(i+1)*shape[1]] = img[:, :, i]

    Image.fromarray(side_by_side).show()
