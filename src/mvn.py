import numpy as np
from numba import njit


@njit
def gaussian_cond_mean(mu, cov, x):
    """Compute the conditional mean of a Gaussian distribution, after observing
    the first n entries of the vector given by x."""
    n = len(x)
    cov11, cov21 = cov[:n, :n], cov[n:, :n]
    gain = np.linalg.solve(cov11.T, cov21.T).T
    return mu[n:] + np.dot(gain, (x - mu[:n]))


@njit
def gaussian_log_density(x, mu, cov):
    k = len(mu)
    L = np.linalg.cholesky(cov)
    I = np.eye(k)
    L_inv = np.linalg.solve(L, I)
    det_cov = np.prod(np.diag(L)) ** 2
    norm_const = -0.5 * (k * np.log(2 * np.pi) + np.log(det_cov))
    diff = x - mu
    y = np.dot(L_inv, diff)
    exponent = -0.5 * np.dot(y.T, y)
    return norm_const + exponent


@njit
def gaussian_density(x, mu, cov):
    k = len(mu)
    L = np.linalg.cholesky(cov)
    I = np.eye(k)
    L_inv = np.linalg.solve(L, I)
    det_cov = np.prod(np.diag(L)) ** 2
    norm_const = 1.0 / (np.power((2 * np.pi), k / 2) * np.sqrt(det_cov))
    diff = x - mu
    y = np.dot(L_inv, diff)
    exponent = -0.5 * np.dot(y.T, y)
    return norm_const * np.exp(exponent)
