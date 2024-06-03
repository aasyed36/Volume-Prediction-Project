import numpy as np
from numba import njit
from sklearn.mixture import GaussianMixture
from .gfm import GaussianFactorMixture


@njit
def gaussian_cond_mean(mu, cov, x):
    """Compute the conditional mean of a Gaussian distribution, after observing
    the first n entries of the vector given by x."""
    n = len(x)
    cov11, cov21 = cov[:n, :n], cov[n:, :n]
    gain = np.linalg.solve(cov11.T, cov21.T).T
    return mu[n:] + np.dot(gain, (x - mu[:n]))


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


class GaussianMixtureVolume:
    """Modeling trading volumes using Gaussian mixtures."""

    def __init__(self, **gmm_args):
        self.gmm_args = gmm_args
        if "rank" in gmm_args:
            self.prior = GaussianFactorMixture(**gmm_args)
        else:
            self.prior = GaussianMixture(**gmm_args)

    @property
    def dim(self):
        return self.prior.means_.shape[1]

    @property
    def weights(self):
        return self.prior.weights_

    @property
    def means(self):
        return self.prior.means_

    @property
    def covariances(self):
        return self.prior.covariances_

    @property
    def n_components(self):
        return self.prior.n_components

    def fit(self, X):
        """Fit the prior distribution of the intraday trading volumes, using
        full days of data."""
        self.prior.fit(X)

    def predict_point(self, x=None):
        """Predict the rest of the vector, conditioned on the first few entries,
        given by x."""
        if x is None:
            return (self.means.T * self.weights).sum(axis=1)

        # observe the first n data points, and predict the remaining entries.
        n = len(x)
        pred_dim = self.dim - n
        assert pred_dim > 0, "Already observed everything -- nothing left to predict."

        # compute modified weights
        weights = np.array(
            [
                gaussian_density(x, self.means[k][:n], self.covariances[k][:n, :n])
                for k in range(self.n_components)
            ]
        )
        weights = self.weights * weights / weights.sum()

        # compute new means and covariances, generate new model
        res = np.zeros(pred_dim)
        for k in range(self.n_components):
            cond_mean_k = gaussian_cond_mean(self.means[k], self.covariances[k], x)
            res += weights[k] * cond_mean_k
        return res

    def predict(self, X=None):
        """Broadcast the predict_point method."""
        if X is None:
            return self.predict_point()
        else:
            return np.array([self.predict_point(x) for x in X])
