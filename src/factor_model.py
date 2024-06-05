"""
Factor models for volume prediction.

* fit_factor_model_kl(cov, rank) implements an EM algorithm (Emmanuel)
* fit_factor_model_frob(cov, rank) implements an ADMM algorithm (Boyd)
* convert_gmm_prior_to_factor_model(gmm) makes the covariance matrices diagonal plus low-rank
"""
import numpy as np
from numba import njit


@njit
def factor_model_heuristic(R, r):
    """
    param R: nxn numpy array, correlation matrix
    param r: float, rank of low rank component

    returns: low rank + diag approximation of R\
        R_hat = sum_i^r lambda_i q_i q_i' + E, where E is diagonal,
        defined so that R_hat has unit diagonal; lamda_i, q_i are eigenvalues
        and eigenvectors of R (the r first, in descending order)
    """
    # ascending order of the largest r eigenvalues
    lamda, Q = np.linalg.eigh(R)
    F = Q[:, -r:] * np.sqrt(lamda[-r:])
    d = np.diag(R - np.dot(F, F.T))
    return d, F


##### Expectation-Maximization algorithm proposed by Emmanuel Candes #####

@njit
def _e_step(Sigma, F, d):
    G = np.linalg.solve(
        (F.T * (1 / d.reshape(1, -1))) @ F + np.eye(F.shape[1]), np.eye(F.shape[1])
    )
    L = G @ F.T * (1 / d.reshape(1, -1))
    Cxx = Sigma
    Cxs = Sigma @ L.T
    Css = L @ (Sigma @ L.T) + G
    return Cxx, Cxs, Css


@njit
def _m_step(Cxx, Cxs, Css):
    F = np.linalg.solve(Css.T, Cxs.T).T
    d = (
        np.diag(Cxx)
        - 2 * np.sum(Cxs * F, axis=1)
        + np.sum(F * (np.dot(F, Css)), axis=1)
    )
    return F, d


@njit
def fit_factor_model_kl(sigma, rank, n_iters=5):
    """
    param sigma: one covariance matrices
    param rank: float, rank of low rank component

    returns: regularized covariance matrices
    """
    d, F = factor_model_heuristic(sigma, rank)
    for _ in range(n_iters):
        Cxx, Cxs, Css = _e_step(sigma, F, d)
        F, d = _m_step(Cxx, Cxs, Css)
    return d, F


##### ADMM algorithm in Frontiers paper #####

@njit
def project_to_psd_cone_factor(X, rank):
    """Project a symmetric matrix X to the cone of positive semidefinite matrices with rank at most k."""
    eigvals, eigvecs = np.linalg.eigh(X)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    eigvals[rank:] = 0.0
    return eigvecs * np.sqrt(eigvals)

@njit
def project_to_psd_cone(X, rank):
    """Project a symmetric matrix X to the cone of positive semidefinite matrices with rank at most k."""
    F = project_to_psd_cone_factor(X, rank)
    return np.dot(F, F.T)


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
def fit_factor_model_frob(
    Sigma,
    rank,
    max_iter=100,
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
    _, F = factor_model_heuristic(Sigma, rank)
    X = np.zeros((n, n))
    Z = F @ F.T
    U = np.zeros((n, n))

    for _ in range(max_iter):
        Z_old = Z.copy()

        X = x_update(Sigma, rho, Z, U, n)
        AX = alpha * X + (1 - alpha) * Z_old
        Z = project_to_psd_cone(AX + U, rank)
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
    F = project_to_psd_cone_factor(Z, rank)
    return d, F


def convert_gmm_prior_to_factor_model(prior, rank, metric="kl"):
    """Convert a Gaussian Mixture Model to a Factor Model."""
    if metric == "kl":
        factorizer = fit_factor_model_kl
    elif metric == "frob":
        factorizer = fit_factor_model_frob
    elif metric == "heuristic":
        factorizer = factor_model_heuristic
    else:
        raise ValueError("metric must be 'kl' or 'frob' or 'heuristic'")

    for k in range(prior.n_components):
        d, F = factorizer(prior.covariances_[k], rank)
        prior.covariances_[k] = np.diag(d) + F @ F.T
    return prior
