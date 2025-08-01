import numpy as np
from utils.dataloader import load_video


def singular_value_thresholding(X, tau):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_thresh = np.maximum(S - tau, 0)
    return U @ np.diag(S_thresh) @ Vt


def rank1_projection(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return S[0] * np.outer(U[:, 0], Vt[0, :])


def soft_thresholding(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def rpca_admm(
        X, lamb=1e-2, rho=1e-3, max_iter=1000, verbose=False, mu=20, tau=20
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


if __name__ == "__main__":
    from utils.visualize import visualize_cube

    x = load_video("./datasets/video/casia_angleview_p01_jump_a1.mp4")[30:60]

    F, M, N = x.shape

    x_bar = x.reshape(F, M * N)

    L, S, rho = rpca_admm(x_bar, lamb=1/np.sqrt(M*N), rho=10)
    print(np.linalg.norm(S, 1))

    L = L.reshape(F, M, N)
    S = S.reshape(F, M, N)
