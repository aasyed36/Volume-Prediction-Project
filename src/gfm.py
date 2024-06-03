import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

from .fit_factor_model import fit_factor_model


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
        n_samples, _ = X.shape
        responsibilities = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            rv = multivariate_normal(self.means_[k], self.covariances_[k])
            responsibilities[:, k] = self.weights_[k] * rv.pdf(X)
        sum_responsibilities = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= sum_responsibilities
        return responsibilities

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        effective_n = responsibilities.sum(axis=0)
        self.weights_ = effective_n / n_samples
        self.means_ = np.dot(responsibilities.T, X) / effective_n[:, np.newaxis]
        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            Sigma = np.dot(responsibilities[:, k] * diff.T, diff) / effective_n[k]
            if self.rank is not None or self.rank == Sigma.shape[0]:
                d, factored = fit_factor_model(Sigma, self.rank, **self.admm_args)
                self.covariances_[k] = np.diag(d) + factored
            else:
                self.covariances_[k] = Sigma

    def fit(self, X):
        self._initialize_parameters(X)
        log_likelihood = None
        for _ in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            new_log_likelihood = np.sum(
                np.log(
                    np.sum(
                        [
                            self.weights_[k]
                            * multivariate_normal(
                                self.means_[k], self.covariances_[k]
                            ).pdf(X)
                            for k in range(self.n_components)
                        ],
                        axis=0,
                    )
                )
            )
            if (
                log_likelihood is not None
                and abs(new_log_likelihood - log_likelihood) < self.tol
            ):
                break
            log_likelihood = new_log_likelihood

    def predict_proba(self, X):
        return self._e_step(X)

    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def score_samples(self, X):
        return np.log(
            np.sum(
                [
                    self.weights_[k]
                    * multivariate_normal(self.means_[k], self.covariances_[k]).pdf(X)
                    for k in range(self.n_components)
                ],
                axis=0,
            )
        )
