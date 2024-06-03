import numpy as np
from numba import njit


@njit
def project_to_psd_cone(X, rank_k):
    """Project a symmetric matrix X to the cone of positive semidefinite matrices with rank at most k."""
    eigvals, eigvecs = np.linalg.eigh(X)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    eigvals[rank_k:] = 0.0
    return (eigvecs * eigvals).dot(eigvecs.T)


@njit
def x_update(Sigma, rho, Z, U, n):
    X = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                X[i, j] = (1 / (1 + rho)) * (Sigma[i, j] + rho * (Z[i, j] - U[i, j]))
            elif Sigma[i, i] - U[i, i] <= Z[i, i]:
                X[i, i] = (1 / (1 + rho)) * (Sigma[i, i] + rho * (Z[i, i] - U[i, i]))
            else:
                X[i, i] = Z[i, i] - U[i, i]
    return X


@njit
def fit_factor_model(
    Sigma,
    k,
    max_iter=1000,
    rho=1.0,
    atol=1e-3,
    rtol=1e-3,
    mu=10,
    tau=2,
    alpha=1.6,
):
    """ADMM algorithm for factor model fitting."""
    n = Sigma.shape[0]

    # Initialize variables
    X = np.zeros((n, n))
    Z = np.zeros((n, n))
    U = np.zeros((n, n))

    for _ in range(max_iter):
        Z_old = Z.copy()

        X = x_update(Sigma, rho, Z, U, n)
        AX = alpha * X + (1 - alpha) * Z_old
        Z = project_to_psd_cone(AX + U, k)
        U = U + AX - Z

        # primal and dual residual
        r = np.linalg.norm(X - Z)
        s = rho * np.linalg.norm(Z - Z_old)

        # Convergence check
        eps_pri = np.sqrt(n) * atol + rtol * max(np.linalg.norm(X), np.linalg.norm(Z))
        eps_dual = np.sqrt(n) * atol + rtol * rho * np.linalg.norm(U)
        if r < eps_pri and s < eps_dual:
            break

        # Update penalty parameter
        if r > mu * s:
            rho *= tau
            U /= tau
        elif s > mu * r:
            rho /= tau
            U *= tau

    d = np.clip(np.diag(Sigma) - np.diag(Z), 0, None)
    return d, Z
