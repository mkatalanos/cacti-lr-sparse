from typing import Any, Callable, Tuple

import numpy as np
from main import init
# from numba import njit
from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_svd
from utils.patches import extract_sparse_patches, reconstruct_sparse_patches
from utils.physics import phi, phit
from utils.visualize import visualize_cube


# @njit
def soft_thresh(x, lambda_):
    return np.sign(x) * np.maximum(np.abs(x)-lambda_, 0)


# @njit
def bar(x: NDArray) -> NDArray:
    # assert len(x.shape) == 3
    M, N, F = x.shape
    x_bar = x.reshape(M*N, F)

    # x_bar = np.zeros((M*N, F))
    #
    # for f in range(F):
    #     x_bar[:, f] = x[:, :, f].reshape(-1)

    return x_bar


# @njit
def update_S(Y, B, mask, U, V, Theta, rho):

    M, N, F = mask.shape

    Y_b = Y-phi(B, mask).reshape((M, N))
    C1 = rho*(U+V-1/rho * Theta)
    C2 = np.zeros((M, N, F))
    for f in range(F):
        C2[:, :, f] = np.multiply(mask[:, :, f], Y_b)

    C3 = C1 + C2
    S = np.zeros((M, N, F))

    mff = np.multiply(mask, mask)

    # for i in range(F):
    S = np.multiply(np.divide(1, mff+2*rho), C3)

    # S[:, :, f] = np.multiply(np.divide(1, mff[:, :, f]+2*rho), C3[:, :, f])

    return S


def update_L(B, Delta, rho, lambda_2, mask, delta=1e-3,
             epsilon=1e-3, max_it=1000, svd_l=10, ):
    M, N, F = mask.shape
    La = B + (1/rho)*Delta
    La_bar = bar(La)
    u, s, vh = randomized_svd(La_bar, svd_l)
    # s is array of components
    dold = s.copy()

    for t in range(max_it):
        d = s-(lambda_2/rho)*(1/(dold+epsilon))
        d = d.clip(0)
        if np.abs(d-dold).max() <= delta:
            break

    L_bar = u @ np.diag(d) @ vh

    L = L_bar.reshape(M, N, F)
    return L


# @njit
def update_U(S, Theta, lambda_1, rho):
    U_a = S + Theta/rho
    return soft_thresh(U_a, lambda_1/rho)


def update_V(S, Gamma, rho, lambda_3,
             max_it=50, epsilon=1e-3, delta=1e-3, svd_l=10, patch_size=16):

    Va = S + Gamma/rho
    Va_tilde, patch_locations = extract_sparse_patches(Va, patch_size)

    u, s, vh = randomized_svd(Va_tilde, svd_l)

    # s is array of components
    dold = s.copy()

    for t in range(max_it):
        d = s-(lambda_3/rho)*(1/(dold+epsilon))
        d = d.clip(0)
        if np.abs(d-dold).max() <= delta:
            break
    V_tilde = u @ np.diag(d) @ vh

    V = reconstruct_sparse_patches(V_tilde, patch_locations, S.shape)

    return V


# @njit
def update_B(Y, mask, S, L, Delta, rho):

    M, N, F = mask.shape

    Ys = Y - phi(S, mask).reshape(M, N)
    C1 = rho*(L-Delta/rho)
    C2 = np.zeros((M, N, F))
    for f in range(F):
        C2[:, :, f] = np.multiply(mask[:, :, f], Ys)

    C3 = C1 + C2
    B = np.zeros((M, N, F))

    mff = np.multiply(mask, mask)

    # for i in range(F):
    B = np.multiply(np.divide(1, mff+rho), C3)

    return B


def ADMM(y, mask, rho=0.8, lambda_1=0.5, lambda_2=0.5, lambda_3=0.5, MAX_IT=3):
    M, N, F = mask.shape

    # Init
    Y = y.reshape((M, N))
    B = np.random.randint(0, 256, (M, N, F))
    U = np.random.randint(0, 256, (M, N, F))
    V = np.random.randint(0, 256, (M, N, F))
    # X = phit(y, mask).reshape(M, N, F)
    # U = X.copy()
    # B = X.copy()
    # V = X.copy()

    B_old = np.zeros_like(B)
    U_old = np.zeros_like(U)
    V_old = np.zeros_like(V)

    Theta = np.random.randint(0, 256, (M, N, F))
    Delta = np.random.randint(0, 256, (M, N, F))
    Gamma = np.random.randint(0, 256, (M, N, F))
    for it in range(MAX_IT):
        print(f"Starting iteration: {it},{rho=}")
        # Can be done in parallel
        # update S
        S = update_S(Y, B, mask, U, V, Theta, rho)
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
        Theta = Theta/rho + S - U
        Gamma = Gamma/rho + S - V
        Delta = Delta/rho + B - L

        Theta *= rho
        Gamma *= rho
        Delta *= rho

        primal_res = np.array([S-U, S-V, B-L])
        dual_res = -rho*np.array([U-U_old + V-V_old, B-B_old])

        primal_res_norm = np.linalg.norm(primal_res)
        dual_res_norm = np.linalg.norm(dual_res)

        # rho = rho * (primal_res_norm/dual_res_norm) ** 0.8

        U_old = U
        V_old = V
        B_old = B

    return S, L, U, V, B


if __name__ == "__main__":
    x, y, mask = init(dataset="./datasets/traffic48_cacti.mat")
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

    S, L, U, V, B = ADMM(y, mask, 0.8, MAX_IT=50)
    visualize_cube((B+S)*255)
    raise SystemExit
