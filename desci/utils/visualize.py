import numpy as np
from PIL import Image


def visualize_cube(img):
    if len(img.shape) != 3:
        img = img[:, np.newaxis]
    assert len(img.shape) == 3

    H, W, F = img.shape
    side_by_side = np.zeros((H, W * F), dtype=np.uint8)

    for i in range(F):
        side_by_side[:, i * W : (i + 1) * W] = img[:, :, i]

    Image.fromarray(side_by_side).show()


def visualize_patches(patches):
    P, L = patches.shape
    patch_size = int(np.sqrt(P))

    assert patch_size * patch_size == P

    visualize_cube(patches.reshape(patch_size, patch_size, -1))


def visualize_clusters(clusters):
    cluster_count = len(clusters)
    if cluster_count == 0:
        return
    P, L = clusters[0].shape
    p = int(np.sqrt(P))

    # Finding smallest common patch count
    T = np.array([cluster.shape for cluster in clusters]).min(axis=0)[1]

    X = np.zeros((p * cluster_count, p * T))
    for row, cluster in enumerate(clusters):
        patches = cluster.reshape(p, p, -1)[:, :, :T]
        for col in range(T):
            patch = patches[:, :, col]
            X[row * p : (row + 1) * p, col * p : (col + 1) * p] = patch
    Image.fromarray(X).show()
