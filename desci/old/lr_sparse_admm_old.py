from typing import Any, Callable, Tuple

import numpy as np

# from numba import njit
from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_svd
from utils.patches import extract_sparse_patches, reconstruct_sparse_patches
from utils.physics import phi, phit, init
from utils.visualize import visualize_cube


# @njit
def soft_thresh(x, lambda_):
    x = x.astype(np.float64)
    out = np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
    return out.clip(0)


# @njit
def bar(x: NDArray) -> NDArray:
    # assert len(x.shape) == 3
    M, N, F = x.shape
    x_bar = x.reshape(M * N, F)

    # x_bar = np.zeros((M*N, F))
    #
    # for f in range(F):
    #     x_bar[:, f] = x[:, :, f].reshape(-1)

    return x_bar


# @njit
def update_S(Y, B, mask, U, V, Theta, Gamma, rho):

    M, N, F = mask.shape

    Y_b = Y - phi(B, mask).reshape((M, N))
    C1 = rho * (U + V - 1 / rho * (Theta + Gamma))
    # C2 = np.zeros((M, N, F))
    # for f in range(F):
    #     C2[:, :, f] = np.multiply(mask[:, :, f], Y_b)
    C2 = np.multiply(mask, Y_b[:, :, np.newaxis])
    C3 = C1 + C2
    S = np.zeros((M, N, F))

    mff = np.multiply(mask, mask)

    # for i in range(F):
    S = np.multiply(np.divide(1, mff + 2 * rho), C3)

    # S[:, :, f] = np.multiply(np.divide(1, mff[:, :, f]+2*rho), C3[:, :, f])

    return S


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


# @njit
def update_U(S, Theta, lambda_1, rho):
    U_a = S + Theta / rho
    return soft_thresh(U_a, lambda_1 / rho)


def update_V(
    S,
    Gamma,
    rho,
    lambda_3,
    max_it=50,
    epsilon=1e-3,
    delta=1e-3,
    svd_l=10,
    patch_size=32,
):

    Va = S + Gamma / rho
    Va_tilde, patch_locations = extract_sparse_patches(Va, patch_size)

    u, s, vh = randomized_svd(Va_tilde, svd_l)

    # s is array of components
    dold = s.copy()

    for t in range(max_it):
        d = s - (lambda_3 / rho) * (1 / (dold + epsilon))
        d = np.maximum(d, 0)
        dold = d
        if np.abs(d - dold).max() <= delta:
            break
    V_tilde = u @ np.diag(d) @ vh

    V = reconstruct_sparse_patches(V_tilde, patch_locations, S.shape)

    return V


# @njit
def update_B(Y, mask, S, L, Delta, rho):

    M, N, F = mask.shape

    Ys = Y - phi(S, mask).reshape(M, N)
    C1 = rho * (L - Delta / rho)
    # C2 = np.zeros((M, N, F))
    # for f in range(F):
    #     C2[:, :, f] = np.multiply(mask[:, :, f], Ys)

    C2 = np.multiply(mask, Ys[:, :, np.newaxis])
    C3 = C1 + C2
    B = np.zeros((M, N, F))

    mff = np.multiply(mask, mask)

    # for i in range(F):
    B = np.multiply(np.divide(1, mff + rho), C3)

    return B


def ADMM(y, mask, rho=0.8, lambda_1=0.8, lambda_2=0.8, lambda_3=0.8, MAX_IT=3):
    M, N, F = mask.shape

    # Init
    Y = y.reshape((M, N)).astype(np.float64)

    U = np.zeros_like(mask, dtype=np.float64)
    B = np.repeat(Y[:, :, np.newaxis], F, axis=2)
    V = np.zeros_like(mask, dtype=np.float64)

    B_old = np.zeros_like(B, dtype=np.float64)
    U_old = np.zeros_like(U, dtype=np.float64)
    V_old = np.zeros_like(V, dtype=np.float64)

    Theta = np.zeros_like(mask, dtype=np.float64)
    Delta = np.zeros_like(mask, dtype=np.float64)
    Gamma = np.zeros_like(mask, dtype=np.float64)

    crits = []
    for it in range(MAX_IT):
        print(f"Starting iteration: {it},rho={float(rho):.2f}")
        # Can be done in parallel
        # update S
        S = update_S(Y, B, mask, U, V, Theta, Gamma, rho)
        # update L
        L = update_L(B, Delta, rho, lambda_2, mask)

        # Wait here if parallelizing

        # Can be done in parallel
        # update U
        U = update_U(S, Theta, lambda_1, rho)
        # update V
        V = update_V(S, Gamma, rho, lambda_3)
        # update B
        B = update_B(Y, mask, S, L, Delta, rho)

        # Wait here if parallelizing

        # Update Dual Variables
        Theta = Theta + rho * (S - U)
        Gamma = Gamma + rho * (S - V)
        Delta = Delta + rho * (B - L)

        primal_res = np.array([S - U, S - V, B - L])
        dual_res = -rho * np.array(
            [V - V_old + B - B_old, U - U_old + V - V_old, B - B_old]
        )

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

        U_old = U.copy()
        V_old = V.copy()
        B_old = B.copy()

        V_t, _ = extract_sparse_patches(V, 16)
        crit = (
            0.5 * np.linalg.matrix_norm(Y - phi(B + S, mask).reshape(M, N)).sum()
            + lambda_1 * np.linalg.matrix_norm(U, ord=1).sum()
            + lambda_2 * np.linalg.norm(bar(L), ord="nuc").sum()
            + lambda_3 * np.linalg.norm(V_t, ord="nuc").sum()
        )

        crits.append(crit)
        print(f"Criterion: {(crit):.2e}\n")
    return S, L, U, V, B, crits


if __name__ == "__main__":
    x, y, mask = init(dataset="./datasets/runner40_cacti.mat")
    M, N, F = mask.shape
    # phi, phit, phiphit = generate_phi(mask)
    # x = x.reshape(-1)
    # y = y.reshape(-1)

    # Testing S run
    # Y = y
    # B = np.random.randint(0, 256, mask.shape)
    # mask = mask
    # U = B.copy()
    # V = B.copy()
    # Theta = B.copy()
    # rho = 0.3
    #
    # S = update_S(Y, B, mask, U, V, Theta, rho)

    # Testing L run
    # Y = y
    # B = np.random.randint(0, 256, mask.shape)
    # Delta = np.random.randint(0, 256, mask.shape)
    # rho = 0.3
    # lambda_2 = 0.5
    # L = update_L(B, Delta, rho, lambda_2, mask, max_it=3)

    # Testing B run
    # Y = y
    # mask = mask
    # S = np.random.randint(0, 256, mask.shape)
    # L = S.copy()
    # Delta = S.copy()
    # rho = 0.3
    #
    # B = update_B(Y, mask, S, L, Delta, rho)

    # Testing V run
    # S = np.random.randint(0, 256, mask.shape)
    # Gamma = np.random.randint(0, 256, mask.shape)
    # rho = 0.3
    # lambda_3 = 0.5
    # V = update_V(S, Gamma, rho, lambda_3, 3)
    (
        y,
        mask,
        rho := 0.8,
        lambda_1 := 0.8,
        lambda_2 := 0.8,
        lambda_3 := 0.8,
        MAX_IT := 3,
    )

    M, N, F = mask.shape

    # Init
    Y = y.reshape((M, N)).astype(np.float64)

    U = np.zeros_like(mask, dtype=np.float64)
    B = np.repeat(Y[:, :, np.newaxis], F, axis=2)
    V = np.zeros_like(mask, dtype=np.float64)

    B_old = np.zeros_like(B, dtype=np.float64)
    U_old = np.zeros_like(U, dtype=np.float64)
    V_old = np.zeros_like(V, dtype=np.float64)

    Theta = np.zeros_like(mask, dtype=np.float64)
    Delta = np.zeros_like(mask, dtype=np.float64)
    Gamma = np.zeros_like(mask, dtype=np.float64)
    S, L, U, V, B, crits = ADMM(
        Y, mask, rho=0.8, lambda_1=0.3, lambda_2=1, lambda_3=1, MAX_IT=50
    )
    # primal_res = np.array([S-U, S-V, B-L])
    # visualize_cube((B+S))
    raise SystemExit
