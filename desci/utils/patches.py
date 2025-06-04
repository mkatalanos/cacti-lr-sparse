from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from skimage.util import view_as_windows


def extract_patches(X: NDArray[np.uint8], patch_size: int) -> NDArray[np.uint8]:
    M, N, F = X.shape
    p = patch_size
    P = p * p

    patches = []

    for f in range(F):
        frame = X[:, :, f]
        frame_patches = view_as_windows(frame, (p, p), step=p)
        # Reshape to (num_patches, P)
        frame_patches = frame_patches.reshape(-1, P)
        patches.append(frame_patches)

    # Concatenate patches from all frames: (L, P)
    all_patches = np.vstack(patches)
    # Transpose to get shape (P, L)
    X_tilde = all_patches.T
    return X_tilde


def reconstruct_from_patches(X_tilde, patch_size, shape: Tuple[int, int, int]):
    p = patch_size
    M, N, F = shape
    P, L = X_tilde.shape

    assert P == p * p, "Patch size mismatch."

    num_patches_per_frame = (M // p) * (N // p)
    assert L == num_patches_per_frame * F, "L does not match expected patch count."

    X_rec = np.zeros((M, N, F), dtype=np.uint8)

    for f in range(F):
        # Get patches for frame f
        start_idx = f * num_patches_per_frame
        end_idx = (f + 1) * num_patches_per_frame
        patches = X_tilde[:, start_idx:end_idx].T  # Shape: (num_patches, P)

        # Reshape each patch to (p, p)
        patches_reshaped = patches.reshape(-1, p, p)

        # Place patches into frame
        idx = 0
        for i in range(0, M, p):
            for j in range(0, N, p):
                X_rec[i:i+p, j:j+p, f] = patches_reshaped[idx]
                idx += 1

    return X_rec


def find_similar(patch: NDArray, patches: NDArray, M: int = 20):
    # Euclidean distance
    distances = np.linalg.norm(patches - patch[:, np.newaxis], axis=0)

    top_idx = distances.argsort()[:M]
    return patches[:, top_idx]
