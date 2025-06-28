import cvxpy as cp
import numpy as np
import numpy.linalg as la

from utils.physics import generate_mask, phi_from_mask
# from lr_sparse_admm import update_X
# Fixed data

M, N, F = 50, 10, 3
rho = 2
lambda_0 = 1

Lambda = np.random.randn(M, N, F)
B = np.random.randn(M, N, F)
V = np.random.randn(M, N, F)

targetx = np.random.randn(M, N, F)
mask = generate_mask(targetx.shape, 0.2)
Y = np.multiply(mask, targetx).sum(axis=2)  # Generate a plausible Y

# ----------------------------------

# Initial form

X1 = cp.Variable((M, N, F))


def H(x: cp.Variable):
    return cp.multiply(mask, x).sum(axis=2)


X_flat = X1.flatten("C")
inner_product = Lambda.flatten() @ (X_flat-B.flatten()-V.flatten())

objective = cp.Minimize(
    0.5*lambda_0*cp.sum_squares(Y-H(X1)) +
    inner_product +
    (rho/2) * cp.sum_squares(X1-B-V)
)

problem1 = cp.Problem(objective)
problem1.solve()

# ----------------
# Updated form

X2 = cp.Variable((M, N, F))

objective = cp.Minimize(
    0.5*lambda_0*cp.sum_squares(Y-H(X2)) +
    (rho/2) * cp.sum_squares(X2-(B+V-Lambda/rho))
)

problem2 = cp.Problem(objective)
problem2.solve()

x1 = X1.value
x2 = X2.value

print(f"{la.norm(x1-x2)=}")

np.testing.assert_allclose(x1, x2)


# ----- Matricized -----
m = [mask[:, :, f].reshape(-1).astype(np.float64) for f in range(F)]

H = np.hstack([np.diag(m[f]) for f in range(F)], dtype=np.float64)

# Check matricized y and phi y are the same
z = np.vstack([targetx[:, :, f].reshape(-1) for f in range(F)]).reshape(-1)
mat_y = (H @ z).reshape(M, N)
meas_y = np.multiply(mask, targetx).sum(axis=2)

np.testing.assert_array_equal(meas_y, mat_y)
np.testing.assert_array_equal(meas_y, Y)


# Check HHt is diagonal

np.testing.assert_equal(
    H @ H.T, np.sum([np.diag(m[f]*m[f]) for f in range(F)], axis=0))

# Check inverse of HtH+rho I/lambda_0

lhs = H.T @ H + np.eye(M*N*F) * rho/lambda_0
rhs = lambda_0/rho * (np.eye(M*N*F) - lambda_0/rho * H.T @
                      la.inv(np.eye(M*N)+lambda_0/rho * H @ H.T) @ H)

# Note: assert_equal fails, but the diff norm is 5.2e-16
np.testing.assert_allclose(la.inv(lhs), rhs)

# Testing update matricized
X_a = (B + V - Lambda/rho)
x_a = X_a.reshape(-1)

# x3 = la.inv(lambda_0 * H.T @ H + rho*np.eye(M*N*F)) @ \
#     (lambda_0 * H.T @ Y.reshape(-1) + rho * x_a)

x3 = la.solve(lambda_0 * H.T @ H + rho*np.eye(M*N*F), 
    lambda_0 * H.T @ Y.reshape(-1) + rho * x_a)

x_np_inv = x3.reshape(M, N, F)

print(f"{la.norm(x_np_inv-x1)=}")


# -------------
# update_X


def update_X(Y, B, V, Lambda, mask, rho, lambda_0):

    M, N, F = mask.shape

    C1 = (rho/lambda_0)*(B+V-Lambda/rho)
    C2 = np.multiply(mask, Y[:, :, np.newaxis])
    C3 = C1+C2
    C4 = np.multiply(mask, C3).sum(axis=2)

    DD = np.multiply(mask, mask).sum(axis=2)

    denom = np.ones((M, N))+DD*lambda_0/rho
    C5 = np.divide(C4, denom)
    C6 = np.multiply(mask, C5[:, :, np.newaxis])

    X = (C3-C6*lambda_0/rho)*lambda_0/rho
    return X


x_update = update_X(Y, B, V, Lambda, mask, rho, lambda_0)

print(f"{la.norm(x1-x_update)=}")

# np.testing.assert_allclose(x1, x3)
