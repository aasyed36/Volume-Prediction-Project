import numpy as np
from numba import njit


@njit
def project_to_psd_cone(X, rank_k):
    """Project a symmetric matrix X to the cone of positive semidefinite matrices with rank at most k."""
    eigvals, eigvecs = np.linalg.eigh(X)
    sorted_indices = np.argsort(eigvals)[::-1]  # Sort eigenvalues in descending order
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Keep only the top k eigenvalues
    eigvals[rank_k:] = 0
    return (eigvecs * eigvals).dot(eigvecs.T)


@njit
def factor_model_fitting(Sigma, k, rho, max_iter=1000, tol=1e-6):
    """ADMM algorithm for factor model fitting."""
    n = Sigma.shape[0]

    # Initialize variables
    X = np.zeros((n, n))
    Z = np.zeros((n, n))
    U = np.zeros((n, n))

    for _ in range(max_iter):
        # X-update
        X_old = X.copy()
        X = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    X[i, j] = (1 / (1 + rho)) * (
                        Sigma[i, j] + rho * (Z[i, j] - U[i, j])
                    )
                elif Sigma[i, i] - U[i, i] <= Z[i, i]:
                    X[i, i] = (1 / (1 + rho)) * (
                        Sigma[i, i] + rho * (Z[i, i] - U[i, i])
                    )
                else:
                    X[i, i] = Z[i, i] - U[i, i]

        # Z-update
        Z = project_to_psd_cone(X + U, k)

        # U-update
        U = U + X - Z

        # Convergence check
        if np.linalg.norm(X - X_old) < tol:
            break

    d = np.clip(np.diag(Sigma) - np.diag(X), 0, None)
    return d, Z
