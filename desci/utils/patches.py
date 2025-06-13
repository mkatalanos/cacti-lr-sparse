from typing import List, Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray
from skimage.util import view_as_windows

# def extract_sparse_patches(X: NDArray[np.uint8], patch_size: int):
#     M, N, F = X.shape
#     p = patch_size
#     P = p * p
#
#     stride = p
#
#     patches = []
#     locations = []
#
#     for f in range(F):
#         frame = X[:, :, f]
#
#         # Extract patches
#         for i in range(0, M-p+1, stride):
#             for j in range(0, N-p+1, stride):
#                 patch = frame[i:i+p, j:j+p]
#             # Check if non-empty
#             # Reshape and add if needed
#                 patches.append(patch.flatten())
#                 locations.append((i, j))
#
#     patches = np.vstack(patches).T
#
#     return patches, locations


# @njit
def extract_sparse_patches(X: NDArray[np.uint8], patch_size: int):
    M, N, F = X.shape
    p = patch_size
    P = p * p

    L = 0
    stride = p

    patches = []
    locations = []

    for f in range(F):
        frame = X[:, :, f]

        # Extract patches
        for i in range(0, M-p+1, stride):
            for j in range(0, N-p+1, stride):
                patch = frame[i:i+p, j:j+p]
            # Check if non-empty
                if np.any(patch):
                    # Reshape and add if needed
                    patches.append(patch.flatten())
                    locations.append((i, j, f))
                    L += 1

    patches_mat = np.zeros((P, L))
    for i, patch in enumerate(patches):
        patches_mat[:, i] = patch
    # patches = np.vstack(patches).T

    return patches_mat, locations


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


def reconstruct_sparse_patches(X_tilde,
                               locations: List[Tuple[int, int, int]],
                               shape: Tuple[int, int, int]):
    M, N, F = shape
    P, L = X_tilde.shape
    p = int(np.sqrt(P))
    # Assuming non-overlapping patches
    X = np.zeros(shape)

    for (patch, (i, j, f)) in zip(X_tilde.T, locations):
        X[i:i+p, j:j+p, f] = patch.reshape(p, p)

    return X


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


if __name__ == '__main__':
    M, N, F = 256, 256, 8
    X = np.random.randint(0, 256, (M, N, F))
    X_Tilde, locations = extract_sparse_patches(X, 16)
