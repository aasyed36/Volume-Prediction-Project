import numpy as np
from numba import njit
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

from .mvn import *
from .fit_factor_model import fit_factor_model


@njit
def logsumexp(x):
    """Compute the logsumexp of an array x."""
    max_x = np.max(x)
    sum_exp = np.sum(np.exp(x - max_x))
    result = max_x + np.log(sum_exp)
    return result


@njit
def gmm_log_likelihood(X, weights, means, covariances):
    res = 0.0
    for i in range(X.shape[0]):
        log_den = np.empty(len(weights))
        for k in range(len(weights)):
            log_den[k] = np.log(weights[k]) + gaussian_log_density(
                X[i], means[k], covariances[k]
            )
        res += logsumexp(log_den)
    return res


@njit
def e_step(X, weights, means, covariances):
    log_response = np.zeros((X.shape[0], len(weights)))
    for i in range(X.shape[0]):
        log_den = np.empty(len(weights))
        for k in range(len(weights)):
            log_den[k] = np.log(weights[k]) + gaussian_log_density(
                X[i], means[k], covariances[k]
            )
        log_response[i] = log_den - logsumexp(log_den)

    return np.exp(log_response)


@njit
def m_step(X, responsibilities, weights, means, covariances):
    n_samples, n_features = X.shape
    effective_n = responsibilities.sum(axis=0)
    weights = effective_n / n_samples
    means = np.dot(responsibilities.T, X) / effective_n[:, np.newaxis]
    covariances = np.zeros((len(weights), n_features, n_features))
    for k in range(len(weights)):
        diff = X - means[k]
        covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / effective_n[k]


class GaussianFactorMixture:
    """TODO: Incorporate factor model fitting step"""

    def __init__(
        self,
        n_components,
        rank=None,
        max_iter=100,
        tol=1e-3,
        random_state=None,
        admm_args=None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.rank = rank
        self.admm_args = admm_args if admm_args is not None else {}
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None

    def _initialize_parameters(self, X):
        np.random.seed(self.random_state)
        kmeans = KMeans(
            n_clusters=self.n_components, random_state=self.random_state
        ).fit(X)
        self.means_ = kmeans.cluster_centers_
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        self.covariances_ = np.array([np.cov(X.T) for _ in range(self.n_components)])

    def _e_step(self, X):
        return e_step(X, self.weights_, self.means_, self.covariances_)

    def _m_step(self, X, responsibilities):
        m_step(X, responsibilities, self.weights_, self.means_, self.covariances_)

        # force the covariance matrices to be diagonal plus low-rank
        for k in range(self.n_components):
            if self.rank is not None or self.rank == self.covariances_.shape[-1]:
                d, factored = fit_factor_model(
                    self.covariances_[k], self.rank, **self.admm_args
                )
                self.covariances_[k] = np.diag(d) + factored

    def fit(self, X):
        self._initialize_parameters(X)
        data_ll = None
        for _ in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            new_data_ll = gmm_log_likelihood(
                X, self.weights_, self.means_, self.covariances_
            )
            if data_ll is not None and abs(new_data_ll - data_ll) < self.tol:
                break
            data_ll = new_data_ll

    def predict_proba(self, X):
        return self._e_step(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X):
        return gmm_log_likelihood(X, self.weights_, self.means_, self.covariances_)
