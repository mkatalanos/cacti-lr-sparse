import numpy as np
import cvxpy as cp

# Test two proximal problems, which should be the same

# --------------------------------------------------
# Fixed data

n = 30  # Dimension of the vectors
rho = 2

Gamma = np.random.randn(n, 1)
S = np.random.randn(n, 1)
Delta = np.random.randn(n, 1)
L = np.random.randn(n, 1)
Lambda = np.random.randn(n, 1)
X = np.random.randn(n, 1)
# --------------------------------------------------

# --------------------------------------------------
# Problem 1: initial form

B1 = cp.Variable((n, 1))
V1 = cp.Variable((n, 1))

objective1 = cp.Minimize(cp.sum_squares(V1) +
                         Gamma.T@(S - V1) +
                         Delta.T@(B1 - L) +
                         Lambda.T@(X - B1 - V1) +
                         (rho/2)*cp.sum_squares(S - V1) +
                         (rho/2)*cp.sum_squares(B1 - L) +
                         (rho/2)*cp.sum_squares(X - B1 - V1))
problem1 = cp.Problem(objective1)
problem1.solve()

B_theory = 0.5*(L + X - V1 + (1/rho)*Lambda - (1/rho)*Delta)
assert np.linalg.norm(B1.value - B_theory.value) < 1e-5
# --------------------------------------------------

# --------------------------------------------------
# Problem 2: after replacing B

V2 = cp.Variable((n, 1))

objective2 = cp.Minimize(cp.sum_squares(V2)
                         + Gamma.T@(S - V2)
                         + 0.5*Delta.T@(X - V2 + (1/rho) *
                                        Lambda - (1/rho)*Delta - L)
                         - 0.5*Lambda.T@(L + (1/rho)*Lambda -
                                         (1/rho)*Delta - X + V2)
                         + (rho/2)*cp.sum_squares(S - V2)
                         + (rho/8)*cp.sum_squares(X - V2 +
                                                  (1/rho)*Lambda - (1/rho)*Delta - L)
                         + (rho/8)*cp.sum_squares(L + (1/rho)
                                                  * Lambda - (1/rho)*Delta - X + V2)
                         )

problem2 = cp.Problem(objective2)
problem2.solve()

assert np.linalg.norm(V1.value - V2.value) < 1e-5
# --------------------------------------------------

# --------------------------------------------------
# Problem 3: after manipulation

V3 = cp.Variable((n, 1))

objective3 = cp.Minimize(cp.sum_squares(V3)
                         + (3*rho/4)*cp.sum_squares(V3)
                         - V3.T@((rho/2)*(X - L) + Gamma + 0.5*(Delta + Lambda)
                                 + rho*S))

problem3 = cp.Problem(objective3)
problem3.solve()

assert np.linalg.norm(V1.value - V3.value) < 1e-5
# --------------------------------------------------

# --------------------------------------------------
# Problem 4: after manipulation

V4 = cp.Variable((n, 1))

Va = (1/3)*(X - L + 2*S + (1/rho)*(Delta + Lambda + 2*Gamma))

objective4 = cp.Minimize(cp.sum_squares(V4)
                         + (3*rho/4)*cp.sum_squares(V4 - Va))

problem4 = cp.Problem(objective4)
problem4.solve()

assert np.linalg.norm(V1.value - V4.value) < 1e-5
# --------------------------------------------------

# --------------------------------------------------
# Problem 5: final format

V5 = cp.Variable((n, 1))

# Previous Va

objective5 = cp.Minimize((2/(3*rho))*cp.sum_squares(V5)
                         + 0.5*cp.sum_squares(V5 - Va))

problem5 = cp.Problem(objective5)
problem5.solve()

assert np.linalg.norm(V1.value - V5.value) < 1e-5
# --------------------------------------------------
