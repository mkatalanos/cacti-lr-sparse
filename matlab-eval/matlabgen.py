
import numpy as np
from utils.dataloader import load_video, load_mat
from utils.physics import generate_mask, phi

from scipy import io

frames = 4
block = 0.5

configs = [(frame, block) for frame in [4, 8, 20] for block in [0.2, 0.5]]


for frames, block in configs:

    START_FRAME = 30
    x = load_video(
        "../desci/datasets/video/casia_angleview_p01_jump_a1.mp4")[
        START_FRAME:START_FRAME+frames,
        :, :
    ]

    # x, _, _ = load_mat("./datasets/drop40_cacti.mat")
    # x = x[:frames]

    mask = generate_mask(x.shape, block)
    y = phi(x, mask)

    data = {
        "orig": x.transpose(1, 2, 0),
        "mask": mask.transpose(1, 2, 0).astype(np.double),
        "meas": y[:, :, np.newaxis]
    }

    io.savemat(f"datagen/casia-f{frames:02}-b{block}.mat", data)
