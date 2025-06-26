import cvxpy as cp
import numpy as np

from utils.physics import generate_mask
# Fixed data

M, N, F = 10, 10, 3
rho = 2
lambda_0 = 1

Lambda = np.random.randn(M, N, F)
B = np.random.randn(M, N, F)
V = np.random.randn(M, N, F)

targetx = np.random.randn(M, N, F)
mask = generate_mask(targetx.shape)
Y = np.multiply(mask, targetx).sum(axis=2)  # Generate a plausible Y

# ----------------------------------

# Initial form

X1 = cp.Variable((M, N, F))


def H(x: cp.Variable):
    return cp.multiply(mask, x).sum(axis=2)


X_flat = X1.flatten("C")
inner_product = Lambda.flatten() @ (X_flat-B.flatten()-V.flatten())

objective = cp.Minimize(
    cp.sum_squares(Y-H(X1)) +
    inner_product +
    (rho/2) * cp.sum_squares(X1-B-V)
)

problem1 = cp.Problem(objective)
problem1.solve()

# ----------------
# Updated form

X2 = cp.Variable((M, N, F))

objective = cp.Minimize(
    cp.sum_squares(Y-H(X2)) +
    (rho/2) * cp.sum_squares(X2-(B+V-Lambda/rho))
)

problem2 = cp.Problem(objective)
problem2.solve()

x1 = X1.value
x2 = X2.value

print(f"{np.linalg.norm(x1-x2)=}")

np.testing.assert_allclose(x1, x2)
