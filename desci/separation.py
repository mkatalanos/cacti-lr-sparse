import numpy as np
from utils.dataloader import load_video, load_mat
from numba import njit


@njit(cache=True, fastmath=True)
def singular_value_thresholding(X, tau):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_thresh = np.maximum(S - tau, 0)
    return U @ np.diag(S_thresh) @ Vt


def rank1_projection(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return S[0] * np.outer(U[:, 0], Vt[0, :])


@njit(cache=True, fastmath=True)
def soft_thresholding(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def rpca_admm(
        X, lamb=1e-2, rho=1e-3, max_iter=1000, verbose=False, mu=10, tau=2
):
    M, N = X.shape

    # Initialize
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)

    Lold = np.zeros_like(X)
    Sold = np.zeros_like(X)

    try:
        for k in range(max_iter):
            # Update L
            L = singular_value_thresholding(X - S + Y / rho, 1 / rho)
            # Update S
            S = soft_thresholding(X - L + Y / rho, lamb / rho)
            # Update Y
            Y = Y + rho * (X - L - S)

            residual = X - L - S
            dual_residual = -rho*(L-Lold + S-Sold)

            primal_res_norm = np.linalg.norm(residual)
            dual_res_norm = np.linalg.norm(dual_residual)

            if primal_res_norm > mu * dual_res_norm:
                rho = tau * rho
            elif dual_res_norm > mu * primal_res_norm:
                rho = rho / tau
            else:
                rho = rho

            if verbose and (k % 1 == 0):
                print(
                    f"Iter {k}: primal_res={primal_res_norm:.2e}, dual_res={dual_res_norm:.2e}, rho={rho:.3f}")

            Sold[:] = S
            Lold[:] = L
    except KeyboardInterrupt:
        return L, S, rho

    return L, S, rho


def t_svt3(Y, tau):
    """
    Tensor Singular Value Thresholding (t-SVT) over the third dimension (axis=2).

    Parameters:
        Y   : numpy.ndarray, shape (n1, n2, n3)
        tau : float, threshold parameter

    Returns:
        D_tau_Y : numpy.ndarray, shape (n1, n2, n3)
    """
    n1, n2, n3 = Y.shape

    # Step 1: FFT along the 3rd dimension
    Y_fft = np.fft.fft(Y, axis=2)
    W_fft = np.zeros_like(Y_fft, dtype=np.complex128)

    # Number of unique frequency slices we need to process
    # For real inputs, frequencies k = 0 .. floor(n3/2) are independent
    half_n3 = n3 // 2 + 1

    # Step 2: SVT on each frontal slice in the Fourier domain
    for k in range(half_n3):
        # Extract the k-th frontal slice (matrix of shape (n1, n2))
        Yk = Y_fft[:, :, k]
        # SVD
        U, S, Vh = np.linalg.svd(Yk, full_matrices=False)
        # Soft-threshold singular values
        S_thresh = np.maximum(S - tau, 0)
        # Reconstruct
        W_fft[:, :, k] = U @ np.diag(S_thresh) @ Vh

    # Step 3: Fill remaining slices by conjugate symmetry
    # for real tensors: F[k] = conj(F[n3 - k]) for k > floor(n3/2)
    for k in range(half_n3, n3):
        W_fft[:, :, k] = np.conj(W_fft[:, :, n3 - k])

    # Step 4: Inverse FFT along the 3rd dimension, keep real part
    D_tau_Y = np.fft.ifft(W_fft, axis=2).real

    return D_tau_Y


@njit(cache=True, fastmath=True)
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


def tensor_rpca_admm(
        X, lamb=1e-2, rho=1e-3, max_iter=1000, verbose=False, mu=10, tau=2
):
    F, M, N = X.shape

    # Initialize
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)

    Lold = np.zeros_like(X)
    Sold = np.zeros_like(X)

    try:
        for k in range(max_iter):
            # Update L
            L = t_svt(X - S + Y / rho, 1 / rho)
            # Update S
            S = soft_thresholding(X - L + Y / rho, lamb / rho)
            # Update Y
            Y = Y + rho * (X - L - S)

            residual = X - L - S
            dual_residual = -rho*(L-Lold + S-Sold)

            primal_res_norm = np.linalg.norm(residual)
            dual_res_norm = np.linalg.norm(dual_residual)

            if primal_res_norm > mu * dual_res_norm:
                rho = tau * rho
            elif dual_res_norm > mu * primal_res_norm:
                rho = rho / tau
            else:
                rho = rho

            if verbose and (k % 1 == 0):
                print(
                    f"Iter {k}: primal_res={primal_res_norm:.2e}, dual_res={dual_res_norm:.2e}, rho={rho:.3f}")

            Sold[:] = S
            Lold[:] = L
    except KeyboardInterrupt:
        return L, S, rho

    return L, S, rho


if __name__ == "__main__":
    from utils.visualize import visualize_cube
    from utils.physics import generate_mask, phi, phit, pseudoinverse

    X = load_video("./datasets/video/casia_angleview_p01_jump_a1.mp4")[30:50]

    F, M, N = X.shape

    mask = generate_mask(X.shape, 0.5)
    y = phi(X, mask)

    # backprojected = phit(y, mask)
    # inverse = pseudoinverse(y, mask)

    L, S, rho = tensor_rpca_admm(X, lamb=0.01, rho=0.1, verbose=True)
    print(np.linalg.norm(S.flatten(), 1))
    visualize_cube(L)
    visualize_cube(S)
    #
    # x_bar = x.reshape(F, M * N)
    #
    # L, S, rho = rpca_admm(x_bar, lamb=1e-5, rho=10, verbose=True)
    # print(np.linalg.norm(S, 1))
    #
    # L = L.reshape(F, M, N)
    # S = S.reshape(F, M, N)
