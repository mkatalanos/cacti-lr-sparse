from typing import Any, Callable, Tuple

import numpy as np

# from numba import njit
from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_svd
from utils.patches import extract_sparse_patches, reconstruct_sparse_patches
from utils.dataloader import load_video
from utils.physics import phi, init, generate_mask
from utils.visualize import visualize_cube
from skimage.metrics import peak_signal_noise_ratio


# @njit
def soft_thresh(x, lambda_):
    x = x.astype(np.float64)
    # JM: I just noticed that Algorithm 3 in the pdf has typos. This is
    # correct. Thank you
    out = np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
    # out = denoise_tv_chambolle(x, 1/lambda_)
    return out


def bar(x: NDArray) -> NDArray:
    # assert len(x.shape) == 3
    M, N, F = x.shape
    x_bar = x.reshape(M * N, F)
    return x_bar


# @njit
def update_X(Y, B, V, Lambda, mask, rho, lambda_0):

    M, N, F = mask.shape

    C1 = rho * (B + V - Lambda / rho)
    C2 = np.multiply(mask, Y[:, :, np.newaxis])
    # JM: **Should be C3 = C1 + lambda_0*C2**
    C3 = C1 + lambda_0 * C2

    mff = lambda_0 * np.multiply(mask, mask)

    X = np.multiply(np.divide(1, mff + rho), C3)

    return X


def update_L(
    B, Delta, rho, lambda_2, mask, delta=1e-3, epsilon=1e-3, max_it=1000, svd_l=10
):
    M, N, F = mask.shape
    La = B + Delta / rho
    La_bar = La.reshape(M * N, F)
    u, s, vh = randomized_svd(La_bar, svd_l)
    # s is array of components
    dold = s.copy()

    # TODO: JM: I would be interested to know how many iterations this takes on
    # average
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
    S = (U+V-(Theta+Gamma)/rho)/2
    # JM: Ok, I should update the document with this operation
    return S
    # S = (U + V - (Theta + Gamma) / rho) / 2
    # return S


# @njit


def update_U(S, Theta, lambda_1, rho):
    U_a = S + Theta / rho
    return soft_thresh(U_a, lambda_1 / rho)


def update_V_B(
    X,
    L,
    S,
    Gamma,
    Lambda,
    Delta,
    rho,
    lambda_3,
    max_it=50,
    epsilon=1e-3,
    delta=1e-3,
    svd_l=20,
    patch_size=4,
):

    Va = (X-L+2*S+(Delta+Lambda+2*Gamma)/rho)/3
    Va_tilde, patch_locations = extract_sparse_patches(
        Va, patch_size, stride_ratio=2)

    u, s, vh = randomized_svd(Va_tilde, svd_l)

    # s is array of components
    dold = s.copy()

    for t in range(max_it):
        d = s - ((2*lambda_3)/(3*rho)) * (1 / (dold + epsilon))
        d = d.clip(0)
        dold = d
        if np.abs(d - dold).max() <= delta:
            break

    V_tilde = u @ np.diag(d) @ vh
    V = reconstruct_sparse_patches(V_tilde, patch_locations, S.shape)
    B = (L + X - V + (Lambda - Delta) / rho) / 2
    return V, B


def ADMM(
        y, mask, rho=0.8, lambda_0=0.5, lambda_1=0.5, lambda_2=0.5, lambda_3=0.5, MAX_IT=3, mu=10, tau=2):
    M, N, F = mask.shape

    # Init
    Y = y.reshape((M, N)).astype(np.float64)

    # JM: Initialization for U and V looks good. We expect them to be near
    # zero. What about initializing B with the commented line (np.repeat)?
    U = np.zeros_like(mask, dtype=np.float64)
    # U = np.repeat(Y[:, :, np.newaxis], F, axis=2)
    B = np.zeros_like(mask, dtype=np.float64)
    # B = np.repeat(Y[:, :, np.newaxis]/F, F, axis=2)
    V = np.zeros_like(mask, dtype=np.float64)

    # JM: Here there's potential for error. Why not make copies of U, V, B?
    U_old = U.copy()
    B_old = B.copy()
    V_old = V.copy()

    Theta = np.zeros_like(mask, dtype=np.float64)
    Delta = np.zeros_like(mask, dtype=np.float64)
    Gamma = np.zeros_like(mask, dtype=np.float64)
    Lambda = np.zeros_like(mask, dtype=np.float64)

    # JM: Just to make memory fixed from the beginning (even though it's a small
    # footprint), I would declare crits as a vector/array of tuples with MAX_IT entries.
    crits = []
    try:
        for it in range(MAX_IT):

            print(f"Starting iteration {it} with rho: {float(rho):.2f}")
            # Can be done in parallel
            X = update_X(Y, B, V, Lambda, mask, rho, lambda_0)
            S = update_S(U, V, Theta, Gamma, rho)
            L = update_L(B, Delta, rho, lambda_2, mask)
            # Wait here if parallelizing

            # JM: do you need V as input to update_V_B?
            # Can be done in parallel
            U = update_U(S, Theta, lambda_1, rho)
            V, B = update_V_B(X, L, S, Gamma, Lambda, Delta, rho, lambda_3)

            # Wait here if parallelizing

            # Update Dual Variables
            Theta = Theta + rho * (S - U)
            Gamma = Gamma + rho * (S - V)
            Delta = Delta + rho * (B - L)
            Lambda = Lambda + rho * (X - B - V)

            # JM: **dual residual hasn't been updated: it's missing one entry; see pdf document**
            primal_res = np.array(
                [S - U,
                 S - V,
                 B - L,
                 X - B - V]
            )
            dual_res = -rho * np.array(
                [V-V_old+B-B_old,
                 U-U_old+V-V_old,
                 B - B_old]
            )

            primal_res_norm = np.linalg.norm(primal_res)
            dual_res_norm = np.linalg.norm(dual_res)

            primal_norms = [np.linalg.norm(term) for term in primal_res]

            # JM: these can be defined as parameters outside the function

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

            data_fidelity = 0.5 * \
                np.linalg.norm(Y - phi(X, mask).reshape(M, N))
            l1_term = np.linalg.norm(U.reshape(-1), 1)
            nuc_l_term = np.linalg.norm(bar(L), ord="nuc")
            nuc_v_term = np.linalg.norm(V_t, ord="nuc")

            crit = (
                data_fidelity,
                l1_term,
                nuc_l_term,
                nuc_v_term,
                primal_norms[0],
                primal_norms[1],
                primal_norms[2],
            )
            crits.append(crit)

            print(f"|Y-H(X)|:\t{data_fidelity:.1e}, |U|_1:\t{l1_term:.1e}")
            print(f"|L|_*:\t\t{nuc_l_term:.1e}, |V|_*:\t{nuc_v_term:.1e}")
            print(
                f"|S-U|: {primal_norms[0]: .1e}, |S-V|: {primal_norms[1]: .1e}, |X-B-V|: {primal_norms[2]: .1e}"
            )

            print()
    except KeyboardInterrupt:
        return X, S, L, U, V, B, crits

    return X, S, L, U, V, B, crits


if __name__ == "__main__":
    x = load_video(
        "./datasets/video/casia_angleview_p01_jump_a1.mp4")[:, :, 30:38]
    mask = generate_mask(x.shape, 0.2)
    y = phi(x, mask)

    lambda_0 = 15
    lambda_1 = 0.17
    lambda_2 = 2.0
    lambda_3 = 0.5

    M, N, F = mask.shape

    X, S, L, U, V, B, crits = ADMM(
        y,
        mask,
        rho=100,
        lambda_0=lambda_0,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        MAX_IT=2000,
    )
    psnr = peak_signal_noise_ratio(x, X, data_range=255)
    print(f"PSNR:", psnr)
    visualize_cube(X)
    # # primal_res = np.array([S-U, S-V, B-L])
    # # visualize_cube((B+S))
    raise SystemExit
