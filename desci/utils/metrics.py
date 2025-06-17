"""
Module contains implementations of SSIM and PSNR
Module copied from:
https://github.com/ucaswangls/cacti/blob/main/cacti/utils/metrics.py

"""

import math

import numpy as np


def compare_psnr(img1, img2, shave_border=0):
    img1 = np.array(img1, dtype=np.float32)
    img2 = np.array(img2, dtype=np.float32)
    height, width = img1.shape[:2]
    img1 = img1[
        shave_border : height - shave_border, shave_border : width - shave_border
    ]
    img2 = img2[
        shave_border : height - shave_border, shave_border : width - shave_border
    ]
    imdff = img1 - img2
    rmse = math.sqrt(np.mean(imdff**2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
