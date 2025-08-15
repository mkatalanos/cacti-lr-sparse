import numpy as np
from utils.dataloader import load_video, load_mat
from utils.physics import generate_mask, phi, pseudoinverse
from utils.patches import extract_sparse_patches, reconstruct_sparse_patches
from numba import njit
import time


@njit(cache=True, fastmath=True)
def singular_value_thresholding(X, tau):
    """
    Singular value thresholding
    Args:
        X: NDArray
        tau: float, thresholding parameter
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_thresh = np.maximum(S - tau, 0)
    return U @ np.diag(S_thresh) @ Vt


@njit(cache=True, fastmath=True)
def t_svt(Y, tau):
    """
    Tensor singular value thresholding along the 1st dimension (axis=0)
    Alg 3 from https://arxiv.org/pdf/1804.03728
    Args:
        Y: NDArray
        tau: float, thresholding parameter
    Returns:
        D_tau_Y: NDArray
    """
    n1, n2, n3 = Y.shape

    # Step 1: FFT along the 1st dimension (axis=0)
    Y_fft = np.fft.fft(Y, axis=0)
    W_fft = np.zeros_like(Y_fft, dtype=np.complex128)
    halfn1 = int(np.ceil((n1 + 1) / 2))

    # Step 2: Matrix SVT on each slice of Y_fft
    for i in range(halfn1):
        U, S, Vh = np.linalg.svd(Y_fft[i, :, :], full_matrices=False)
        S_thresh = np.maximum(S - tau, 0)
        W_fft[i, :, :] = (U * S_thresh) @ Vh

    # Fill remaining slices using conjugate symmetry
    for i in range(halfn1, n1):
        W_fft[i, :, :] = np.conj(W_fft[n1 - i, :, :])

    # Step 3: Inverse FFT along the 1st dimension (axis=0)
    D_tau_Y = np.fft.ifft(W_fft, axis=0).real

    return D_tau_Y


@njit(cache=True, fastmath=True)
def soft_thresholding(x, lambda_):
    """
    Soft thresholding operator
    Args:
        x: NDArray
        lambda_: float, thresholding parameter
    """
    x = x.astype(np.float64)
    out = np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
    return out


@njit(cache=True, fastmath=True)
def update_B(X, S, Lambda, lamb1, rho):
    Ba = (X-S+Lambda/rho)
    Ba = np.ascontiguousarray(Ba)  # Needed by Numba
    return t_svt(Ba, lamb1/(2*rho))


@njit(cache=True, fastmath=True)
def update_X(Y, B, S, Lambda, mask, rho, lambda_0):

    F, M, N = mask.shape

    C1 = (rho/lambda_0)*(B+S-Lambda/rho)
    C2 = np.multiply(mask, Y[np.newaxis, :, :])
    C3 = C1+C2
    C4 = np.multiply(mask, C3).sum(axis=0)

    DD = np.multiply(mask, mask).sum(axis=0)

    denom = np.ones((M, N))+DD*lambda_0/rho
    C5 = np.divide(C4, denom)
    C6 = np.multiply(mask, C5[np.newaxis, :, :])

    X = (C3-C6*lambda_0/rho)*lambda_0/rho
    return X.clip(0, 255)


def solve_admm(
        Y, mask, lamb0, lamb1, lamb2, rho=1, max_iter=1000, verbose=False, mu=10, tau=2
):
    F, M, N = mask.shape

    # Initialize
    # B = np.zeros_like(mask)
    B = pseudoinverse(y, mask)
    S = np.zeros_like(mask)
    Lambda = np.zeros_like(mask)

    Bold = np.random.randn(*mask.shape)
    Sold = np.random.randn(*mask.shape)

    try:
        for k in range(max_iter):
            # Update X
            X = update_X(Y, B, S, Lambda, mask, rho, lamb0)
            # Update B
            B = update_B(X, S, Lambda, lamb1, rho)
            # B = singular_value_thresholding(
            #     (X-S+Lambda/rho).reshape(F, M*N), lamb1/(2*rho)).reshape(F, M, N)
            # Update S
            S = soft_thresholding(X-B+Lambda/rho, lamb2/(2*rho))

            # Update Lambda
            Lambda = Lambda + rho * (X - B - S)

            residual = X - B - S
            dual_residual = -rho*(B-Bold + S-Sold)

            primal_res_norm = np.linalg.norm(residual)
            dual_res_norm = np.linalg.norm(dual_residual)

            if k >= 5:
                if primal_res_norm > mu * dual_res_norm:
                    rho = tau * rho
                elif dual_res_norm > mu * primal_res_norm:
                    rho = rho / tau
                else:
                    rho = rho

            if verbose and (k % 1 == 0):
                Bs = np.linalg.norm(B.reshape(F, M*N), "nuc")
                S1 = np.linalg.norm(S.flatten(), 1)
                fid = np.linalg.norm(Y-phi(X, mask))
                print(
                    f"Iter {k}: primal_res={primal_res_norm:.2e}, dual_res={dual_res_norm:.2e}, rho={rho:.3f}, |B|_*={Bs:.2e}, |S|_1={S1:.2e}, fid={fid:.2e}")

            Sold[:] = S
            Bold[:] = B
    except KeyboardInterrupt:
        return X, B, S, rho

    return X, B, S, rho


if __name__ == "__main__":
    from utils.visualize import visualize_cube
    from skimage.metrics import peak_signal_noise_ratio

    F = 4
    START_FRAME = 30
    x = load_video(
        "./datasets/video/casia_angleview_p01_jump_a1.mp4")[
        START_FRAME:START_FRAME+F,
        :, :
    ]
    mask = generate_mask(x.shape, 0.4)
    y = phi(x, mask)
    F, M, N = mask.shape

    lamb0 = 1
    lamb1 = 1
    lamb2 = 4e-2
    rho = 1
    start = time.time()
    X, B, S, rho = solve_admm(y, mask, lamb0, lamb1, lamb2, rho,
                              max_iter=500, verbose=True)
    duration = time.time()-start
    psnr = peak_signal_noise_ratio(x, X, data_range=255)
    print(f"PSNR: {psnr:.2f}")
    print(f"Duration: {duration:.2f}")
    visualize_cube(X)
