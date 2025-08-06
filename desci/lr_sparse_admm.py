from typing import Any, Callable, Tuple

import numpy as np

from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_svd
from utils.patches import extract_sparse_patches, reconstruct_sparse_patches
from utils.dataloader import load_video, load_mat
from utils.physics import phi, init, generate_mask, pseudoinverse
from utils.visualize import visualize_cube
from skimage.metrics import peak_signal_noise_ratio

from utils.wnnm import framewise_wnnm

from numba import njit


@njit
def soft_thresh(x, lambda_):
    x = x.astype(np.float64)
    out = np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
    return out


@njit
def bar(x: NDArray) -> NDArray:
    # assert len(x.shape) == 3
    F, M, N = x.shape
    x_bar = x.reshape(F, M * N)
    return x_bar


@njit
def update_X(Y, B, V, Lambda, mask, rho, lambda_0):

    F, M, N = mask.shape

    C1 = (rho/lambda_0)*(B+V-Lambda/rho)
    C2 = np.multiply(mask, Y[np.newaxis, :, :])
    C3 = C1+C2
    C4 = np.multiply(mask, C3).sum(axis=0)

    DD = np.multiply(mask, mask).sum(axis=0)

    denom = np.ones((M, N))+DD*lambda_0/rho
    C5 = np.divide(C4, denom)
    C6 = np.multiply(mask, C5[np.newaxis, :, :])

    X = (C3-C6*lambda_0/rho)*lambda_0/rho
    return X


@njit
def t_svt(Y, tau):
    """
    Tensor Singular Value Thresholding (t-SVT) over the first dimension (axis=0)

    Parameters:
        Y   : numpy.ndarray, shape (n1, n2, n3)
        tau : float, threshold parameter

    Returns:
        D_tau_Y : numpy.ndarray, shape (n1, n2, n3)
    """
    n1, n2, n3 = Y.shape
    # Step 1: FFT along the 1st dimension (axis=0)
    Y_fft = np.fft.fft(Y, axis=0)
    W_fft = np.zeros_like(Y_fft, dtype=np.complex128)
    halfn1 = int(np.ceil((n1 + 1) / 2))

    # Step 2: Matrix SVT on each lateral slice (fix i, vary j,k)
    for i in range(halfn1):
        U, S, Vh = np.linalg.svd(Y_fft[i, :, :], full_matrices=False)
        S_thresh = np.maximum(S - tau, 0)
        W_fft[i, :, :] = (U * S_thresh) @ Vh

    # Fill the remaining slices using conjugate symmetry
    for i in range(halfn1, n1):
        W_fft[i, :, :] = np.conj(W_fft[n1 - i, :, :])

    # Step 3: Inverse FFT along the 1st dimension (axis=0)
    # Take real part if input is real
    D_tau_Y = np.fft.ifft(W_fft, axis=0).real

    return D_tau_Y


def update_L_tsvd(
    B, Delta, rho, lambda_2, mask,
    delta=1e-3, epsilon=1e-3, max_it=1000, svd_l=60
):
    """
    Uses T-SVD for background
    """

    F, M, N = mask.shape
    La = B + Delta / rho

    L = t_svt(La, lambda_2/rho)

    return L


@njit
def update_L(
    B, Delta, rho, lambda_2, mask,
    delta=1e-3, epsilon=1e-3, max_it=1000, svd_l=60
):
    """
    L <- B
    """
    F, M, N = mask.shape
    La = B + Delta / rho
    # La_bar = La.reshape(F, M * N)
    # u, s, vh = np.linalg.svd(La_bar, full_matrices=False)
    # u, s, vh = randomized_svd(La_bar, svd_l)
    La_tilde, patch_locations = extract_sparse_patches(La, 32,8)
    u, s, vh = np.linalg.svd(La_tilde, full_matrices=False)

    """ Trying without weighting

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

    """
    d = soft_thresh(s, lambda_2/rho)
    # L_bar = u @ np.diag(d) @ vh
    # L_bar = u[:, :1] @ np.diag(d[:1]) @ vh[:1, :]
    # L = L_bar.reshape(F, M, N)
    L_tilde = u @ np.diag(d) @ vh
    L = reconstruct_sparse_patches(L_tilde, patch_locations, La.shape)
    return L


@njit
def update_S(U, V, Theta, Gamma, rho):
    S = (U+V-(Theta+Gamma)/rho)/2
    return S


# @njit


@njit
def update_U(S, Theta, lambda_1, rho):
    U_a = S + Theta / rho
    return soft_thresh(U_a, lambda_1 / rho)


def update_V_B_fwise(
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
    svd_l=50,
    patch_size=8,
):

    F, M, N = X.shape

    Va = (X-L+2*S+(Delta+Lambda+2*Gamma)/rho)/3
    Va_t = Va.transpose(1, 2, 0)
    c = 2*lambda_3/(rho*3)
    V = np.zeros_like(Va)

    # Framewise denoise
    for f in range(F):
        V[f, :, :] = np.squeeze(
            framewise_wnnm(
                Va_t[:, :, f][:, :, np.newaxis],
                patchRadius=patch_size)
        )

    # V = V_tilde.reshape(F, M, N)
    B = (L + X - V + (Lambda - Delta) / rho) / 2
    return V, B


@njit
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
    svd_l=50,
    patch_size=8,
):

    F, M, N = X.shape

    Va = (X-L+2*S+(Delta+Lambda+2*Gamma)/rho)/3

    Va_bar = bar(Va)
    u, s, vh = np.linalg.svd(Va_bar, full_matrices=False)

    """ Testing without weighing
    # s is array of components
    dold = s.copy()

    for t in range(max_it):
        d = s - ((2*lambda_3)/(3*rho)) * (1 / (dold + epsilon))
        d = d.clip(0)
        dold = d
        if np.abs(d - dold).max() <= delta:
            break
"""
    d = soft_thresh(s, 2*lambda_3/(rho*3))
    V_tilde = u @ np.diag(d) @ vh
    V = V_tilde.reshape(F, M, N)
    B = (L + X - V + (Lambda - Delta) / rho) / 2
    return V, B


def update_V_B_patches(
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
    svd_l=50,
    patch_size=8,
):

    F, M, N = X.shape

    Va = (X-L+2*S+(Delta+Lambda+2*Gamma)/rho)/3
    Va_tilde, patch_locations = extract_sparse_patches(
        Va, patch_size, stride_ratio=16)

    u, s, vh = randomized_svd(Va_tilde, svd_l)
    u, s, vh = np.linalg.svd(Va_tilde, full_matrices=False)

    # Va_bar = bar(Va)
    # u, s, vh = np.linalg.svd(Va_bar, full_matrices=False)

    """ Testing without weighing
    # s is array of components
    dold = s.copy()

    for t in range(max_it):
        d = s - ((2*lambda_3)/(3*rho)) * (1 / (dold + epsilon))
        d = d.clip(0)
        dold = d
        if np.abs(d - dold).max() <= delta:
            break
"""
    d = soft_thresh(s, 2*lambda_3/(rho*3))
    V_tilde = u @ np.diag(d) @ vh
    # V = t_svt(Va, 2*lambda_3/(rho*3))
    V = reconstruct_sparse_patches(V_tilde, patch_locations, S.shape)
    # V = V_tilde.reshape(F, M, N)
    B = (L + X - V + (Lambda - Delta) / rho) / 2
    return V, B


def ADMM(
        y, mask, rho=0.8, lambda_0=0.5, lambda_1=0.5, lambda_2=0.5, lambda_3=0.5, MAX_IT=3, mu=10, tau=2, verbose=True):
    F, M, N = mask.shape

    # Init
    Y = y.reshape((M, N)).astype(np.float64)

    # JM: Initialization for U and V looks good. We expect them to be near
    # zero. What about initializing B with the commented line (np.repeat)?
    U = np.zeros_like(mask, dtype=np.float64)
    # U = np.repeat(Y[:, :, np.newaxis], F, axis=2)
    B = pseudoinverse(Y, mask)
    # B = np.repeat(Y[:, :, np.newaxis]/F, F, axis=2)
    # B = np.zeros_like(mask, dtype=np.float64)
    V = np.zeros_like(mask, dtype=np.float64)

    # JM: Here there's potential for error. Why not make copies of U, V, B?
    U_old = U.copy()
    B_old = B.copy()
    V_old = V.copy()

    Theta = np.random.randn(*mask.shape)
    Delta = np.random.randn(*mask.shape)
    Gamma = np.random.randn(*mask.shape)
    Lambda = np.random.randn(*mask.shape)
    # Delta = np.zeros_like(mask, dtype=np.float64)
    # Gamma = np.zeros_like(mask, dtype=np.float64)
    # Lambda = np.zeros_like(mask, dtype=np.float64)

    # JM: Just to make memory fixed from the beginning (even though it's a small
    # footprint), I would declare crits as a vector/array of tuples with MAX_IT entries.
    crits = []
    try:
        for it in range(MAX_IT):

            if verbose:
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

            # if rho < 0.01:
            #     print(f"value of {rho=:.2e} too small")
            #     break
            if primal_res_norm < 1e-4 and dual_res_norm<1e-4:
                if verbose:
                    print(f"{primal_res_norm=:.2e} or {dual_res_norm=:.2e} less than 1e-4")
                break

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

            if verbose:
                print(f"|Y-H(X)|:\t{data_fidelity:.1e}, |U|_1:\t{l1_term:.1e}")
                print(f"|L|_*:\t\t{nuc_l_term:.1e}, |V|_*:\t{nuc_v_term:.1e}")
                print(
                    f"|S-U|: {primal_norms[0]: .1e}, |S-V|: {primal_norms[1]
                        : .1e}, |X-B-V|: {primal_norms[2]: .1e}"
                )

                print()
    except KeyboardInterrupt:
        return X, S, L, U, V, B, crits

    return X, S, L, U, V, B, crits


if __name__ == "__main__":
    F = 20
    START_FRAME = 30
    x = load_video(
        "./datasets/video/casia_angleview_p01_jump_a1.mp4")[
        START_FRAME:START_FRAME+F,
        :, :
    ]
    mask = generate_mask(x.shape, 0.5)
    y = phi(x, mask)

    # x, mask, y = load_mat("./datasets/drop40_cacti.mat")

    F, M, N = mask.shape

    # lambda_0 = 1
    # lambda_1 = 0.1
    # lambda_2 = 30.0
    # lambda_3 = 0.5

    lambda_0 = 1
    lambda_1 = 3e-2
    lambda_2 = 10
    lambda_3 = 1e-1
    rho = 1e-2

    X, S, L, U, V, B, crits = ADMM(
        y,
        mask,
        rho=rho,
        lambda_0=lambda_0,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        MAX_IT=500,
    )
    psnr = peak_signal_noise_ratio(x, X, data_range=255)
    print(f"PSNR:", psnr)
    visualize_cube(X)
    # # primal_res = np.array([S-U, S-V, B-L])
    # # visualize_cube((B+S))
    raise SystemExit
