from typing import Any, Callable, Tuple

import numpy as np

# from numba import njit
from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_svd
from utils.patches import extract_sparse_patches, reconstruct_sparse_patches
from utils.dataloader import load_video
from utils.physics import phi, init, generate_mask
from utils.visualize import visualize_cube
from skimage.restoration import denoise_tv_chambolle


# @njit
def soft_thresh(x, lambda_):
    x = x.astype(np.float64)
    out = np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
    # out = denoise_tv_chambolle(x, 1/lambda_)
    return out


def bar(x: NDArray) -> NDArray:
    # assert len(x.shape) == 3
    M, N, F = x.shape
    x_bar = x.reshape(M * N, F)
    return x_bar


# @njit
def update_X(Y, B, V, Lambda, mask, rho):

    M, N, F = mask.shape

    C1 = rho * (B+V-Lambda/rho)
    C2 = np.multiply(mask, Y[:, :, np.newaxis])
    C3 = C1 + C2

    mff = np.multiply(mask, mask)

    X = np.multiply(np.divide(1, mff + rho), C3)

    return X


def update_L(
    B, Delta, rho, lambda_2, mask, delta=1e-3, epsilon=1e-3, max_it=1000, svd_l=30
):
    M, N, F = mask.shape
    La = B + Delta / rho
    La_bar = La.reshape(M * N, F)
    u, s, vh = randomized_svd(La_bar, svd_l)
    # s is array of components
    dold = s.copy()

    for t in range(max_it):
        d = s - (lambda_2 / rho) * (1 / (dold + epsilon))
        d = np.maximum(d, 0)
        dold = d
        if np.abs(d - dold).max() <= delta:
            break

    L_bar = u @ np.diag(d) @ vh

    L = L_bar.reshape(M, N, F)
    return L


def update_S(U, V, Theta, Gamma, rho):
    return (U+V-(Theta+Gamma)/rho)/2
# @njit


def update_U(S, Theta, lambda_1, rho):
    U_a = S + Theta / rho
    return soft_thresh(U_a, lambda_1 / rho)


def update_V_B(X, L, V, S, Gamma, Lambda, Delta, rho, lambda_3, max_it=50, epsilon=1e-3, delta=1e-3, svd_l=10, patch_size=4):

    Va = ((Gamma+Lambda)/rho + S + 0.5*(X-L))/3
    Va_tilde, patch_locations = extract_sparse_patches(
        Va, patch_size, stride_ratio=2)

    u, s, vh = randomized_svd(Va_tilde, svd_l)

    # s is array of components
    dold = s.copy()

    for t in range(max_it):
        d = s - lambda_3 * (1 / (dold + epsilon))
        d = d.clip(0)
        dold = d
        if np.abs(d - dold).max() <= delta:
            break

    V_tilde = u @ np.diag(d) @ vh
    V = reconstruct_sparse_patches(V_tilde, patch_locations, S.shape)
    B = (L+X-V+(Lambda-Delta)/rho)/2
    return V, B


def ADMM(y, mask, rho=0.8, lambda_1=0.8, lambda_2=0.8, lambda_3=0.8, MAX_IT=3):
    M, N, F = mask.shape

    # Init
    Y = y.reshape((M, N)).astype(np.float64)

    U = np.zeros_like(mask, dtype=np.float64)
    # U = np.repeat(Y[:, :, np.newaxis], F, axis=2)
    B = np.zeros_like(mask, dtype=np.float64)
    # B = np.repeat(Y[:, :, np.newaxis]/F, F, axis=2)
    V = np.zeros_like(mask, dtype=np.float64)

    U_old = np.zeros_like(U, dtype=np.float64)
    B_old = np.zeros_like(B, dtype=np.float64)
    V_old = np.zeros_like(V, dtype=np.float64)

    Theta = np.zeros_like(mask, dtype=np.float64)
    Delta = np.zeros_like(mask, dtype=np.float64)
    Gamma = np.zeros_like(mask, dtype=np.float64)
    Lambda = np.zeros_like(mask, dtype=np.float64)

    crits = []
    try:
        for it in range(MAX_IT):

            print(f"Starting iteration: {it},rho={float(rho):.2f}")
            # Can be done in parallel
            X = update_X(Y, B, V, Lambda, mask, rho)
            S = update_S(U, V, Theta, Gamma, rho)
            L = update_L(B, Delta, rho, lambda_2, mask)
            # Wait here if parallelizing

            # Can be done in parallel
            U = update_U(S, Theta, lambda_1, rho)
            V, B = update_V_B(X, L, V, S, Gamma, Lambda, Delta, rho, lambda_3)

            # Wait here if parallelizing

            # Update Dual Variables
            Theta = Theta + rho * (S - U)
            Gamma = Gamma + rho * (S - V)
            Delta = Delta + rho * (B - L)
            Lambda = Lambda + rho * (X-B-V)

            primal_res = np.array([S - U, S - V, B - L, X-B-V])
            dual_res = -rho * np.array([U - U_old + V - V_old, B - B_old])

            primal_res_norm = np.linalg.norm(primal_res)
            dual_res_norm = np.linalg.norm(dual_res)

            mu = 10
            tau = 2

            if primal_res_norm > mu * dual_res_norm:
                rho = tau * rho
            elif dual_res_norm > mu * primal_res_norm:
                rho = rho / tau
            else:
                rho = rho
            # rho=rho*(primal_res_norm/dual_res_norm)

            U_old = U.copy()
            V_old = V.copy()
            B_old = B.copy()

            V_t, _ = extract_sparse_patches(V, 16)
            crit = (
                0.5 * np.linalg.matrix_norm(Y -
                                            phi(X, mask).reshape(M, N)).sum()
                + lambda_1 * np.linalg.matrix_norm(U, ord=1).sum()
                + lambda_2 * np.linalg.norm(bar(L), ord="nuc").sum()
                + lambda_3 * np.linalg.norm(V_t, ord="nuc").sum()
            )

            crits.append(crit)
            print(f"Criterion: {(crit):.2e}")
            # print(f"Primal-dual norm ratio: {(primal_res_norm/dual_res_norm):.2f}")
            print()
    except KeyboardInterrupt:
        return X, S, L, U, V, B, crits

    return X, S, L, U, V, B, crits


if __name__ == "__main__":
    x = load_video(
        "./datasets/video/casia_angleview_p01_jump_a1.mp4")[:, :, 30:90]
    mask = generate_mask(x.shape, 0.5)
    y = phi(x, mask)

    M, N, F = mask.shape

    X, S, L, U, V, B, crits = ADMM(
        y, mask, rho=1, lambda_1=0.5, lambda_2=0.5, lambda_3=0.5, MAX_IT=2000
    )
    visualize_cube(X)
    # # primal_res = np.array([S-U, S-V, B-L])
    # # visualize_cube((B+S))
    raise SystemExit
