import cvxpy as cp
import numpy as np
import numpy.linalg as la

from utils.physics import generate_mask, phi_from_mask
from lr_sparse_admm import soft_thresh

# Fixed data

M, N, F = 50, 10, 3
rho = 10
lambda_1 = 3

Theta = np.random.randn(F, M, N) * 255
S = np.random.randn(F, M, N) * 255
U = np.random.randn(F, M, N) * 255

# ----------------------------------

# Initial form

U1 = cp.Variable((F, M, N))

inner_product = Theta.flatten("F") @ (S.flatten("F") - U1.flatten("F"))

objective = cp.Minimize(
    lambda_1 * cp.norm1(U1) +
    inner_product +
    (rho/2) * cp.sum_squares(S-U1)
)

problem1 = cp.Problem(objective)
problem1.solve()

# ----------------
# Updated form

U2 = cp.Variable((F, M, N))

objective = cp.Minimize(
    lambda_1/rho * cp.norm1(U2) +
    0.5 * cp.sum_squares(U2-(S+Theta/rho))
)

problem2 = cp.Problem(objective)
problem2.solve()

u1 = U1.value
u2 = U2.value

print(f"{la.norm(u1-u2)=}")

# -----
# Soft thresholding solution

u3 = soft_thresh(S + Theta / rho, lambda_1 / rho)

print(f"{la.norm(u1-u3)=}")
