import cvxpy as cp
import numpy as np
import numpy.linalg as la

from utils.physics import generate_mask, phi_from_mask
# from lr_sparse_admm import update_X
# Fixed data

M, N, F = 50, 10, 3
rho = 10
lambda_0 = 3

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


X_flat = X1.flatten("F")
inner_product = Lambda.flatten("F") @ (X_flat-B.flatten("F")-V.flatten("F"))

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
m = [mask[:, :, f].flatten("F").astype(np.float64) for f in range(F)]

H = np.vstack([np.diag(m[f]) for f in range(F)], dtype=np.float64)

# Check matricized y and phi y are the same
z = np.hstack([targetx[:, :, f].flatten("F") for f in range(F)]).flatten("F")
mat_y = (z @ H).reshape(N, M).T
meas_y = np.multiply(mask, targetx).sum(axis=2)

np.testing.assert_array_equal(meas_y, mat_y)
np.testing.assert_array_equal(meas_y, Y)


# Check HHt is diagonal

np.testing.assert_equal(
    H.T @ H, np.sum([np.diag(m[f]*m[f]) for f in range(F)], axis=0))

# Check inverse of HtH+rho I/lambda_0

lhs = H @ H.T + np.eye(M*N*F) * rho/lambda_0
rhs = lambda_0/rho * (
    np.eye(M*N*F)-(lambda_0/rho) *
    H @ la.inv(np.eye(M*N) + H.T @ H * lambda_0/rho) @ H.T
)


# rhs = lambda_0/rho * (np.eye(M*N*F) - lambda_0/rho * H.T @
#                       la.inv(np.eye(M*N)+lambda_0/rho * H @ H.T) @ H)

# Note: assert_equal fails, but the diff norm is 5.2e-16
np.testing.assert_allclose(la.inv(lhs), rhs)


# Testing update matricized
X_a = (B + V - Lambda/rho)
x_a = X_a.flatten("F")

# x3 = la.inv(lambda_0 * H.T @ H + rho*np.eye(M*N*F)) @ \
#     (lambda_0 * H.T @ Y.flatten("F") + rho * x_a)

x3 = la.solve(lambda_0 * H @ H.T + rho*np.eye(M*N*F),
              lambda_0 * Y.flatten("F") @ H.T + rho * x_a)


print(f"{la.norm(x1.flatten("F") - x3)=}")


# -------------
# update_X


def update_X(Y, B, V, Lambda, mask, rho, lambda_0):

    M, N, F = mask.shape
    mask = mask.transpose(2, 0, 1)

    C1 = (rho/lambda_0)*(B+V-Lambda/rho).transpose(2, 0, 1)
    C2 = np.multiply(mask, Y[np.newaxis, :, :])
    C3 = C1+C2
    C4 = np.multiply(mask, C3).sum(axis=0)

    DD = np.multiply(mask, mask).sum(axis=0)

    denom = np.ones((M, N))+DD*lambda_0/rho
    C5 = np.divide(C4, denom)
    C6 = np.multiply(mask, C5[np.newaxis, :, :])

    X = (C3-C6*lambda_0/rho)*lambda_0/rho
    return X.transpose(1, 2, 0)


x_update = update_X(Y, B, V, Lambda, mask, rho, lambda_0)

print(f"{la.norm(x1-x_update)=}")

# np.testing.assert_allclose(x1, x3)
