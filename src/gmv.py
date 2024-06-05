import numpy as np
from numba import njit

from .mvn import gaussian_cond_mean, gaussian_density


@njit
def conditional_weights(x, means, covariances, weights):
    n = len(x)
    cond_weights = np.empty(len(weights))
    for k in range(len(weights)):
        cond_weights[k] = gaussian_density(x, means[k, :n], covariances[k, :n, :n])

    return weights * cond_weights / cond_weights.sum()


@njit
def predict_component(x, weights, means, covariances):
    component = 0
    max_prob = 0.0
    for k in range(len(weights)):
        probk = weights[k] * gaussian_density(x, means[k], covariances[k])
        if probk > max_prob:
            max_prob = probk
            component = k

    return component


class GaussianMixtureVolume:
    """Modeling trading volumes using Gaussian mixtures."""

    def __init__(self, prior):
        # prior needs to have the following attributes: 
        # means_, covariances_, weights_, n_components
        #
        # for example, prior = GaussianMixture(n_components=2) # (scikit-learn)
        # in that case, call prior.fit(X) to fit the model before passing it to
        # this class constructor.
        self.prior = prior

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
        cov = self.prior.covariances_
        if cov.ndim == 2:
            dim = cov.shape[1]
            cov = np.concatenate(
                [
                    np.diag(cov[i]).reshape(1, dim, dim)
                    for i in range(self.n_components)
                ],
                axis=0,
            )
        return cov

    @property
    def n_components(self):
        return self.prior.n_components

    def predict_point(self, x=None, mix_components=False):
        """Predict the rest of the vector, conditioned on the first few entries,
        given by x."""
        if x is None:
            return (self.means.T * self.weights).sum(axis=1)

        # observe the first n data points, and predict the remaining entries.
        n = len(x)
        pred_dim = self.dim - n
        assert pred_dim > 0, "Already observed everything -- nothing left to predict."

        # predict which component is active using past data, then take its mean
        if not mix_components and x is not None:
            k = predict_component(
                x[:n], self.weights, self.means[:, :n], self.covariances[:, :n, :n]
            )
            return gaussian_cond_mean(self.means[k], self.covariances[k], x)

        # compute full conditional mean, mixing the components together
        else:
            # compute modified weights
            weights = conditional_weights(x, self.means, self.covariances, self.weights)

            # compute new means and covariances, generate new model
            res = np.zeros(pred_dim)
            for k in range(self.n_components):
                cond_mean_k = gaussian_cond_mean(self.means[k], self.covariances[k], x)
                res += weights[k] * cond_mean_k

            return res

    def predict(self, X=None, mix_components=False):
        """Broadcast the predict_point method."""
        if X is None:
            return self.predict_point(mix_components=mix_components)
        else:
            return np.array(
                [self.predict_point(x, mix_components=mix_components) for x in X]
            )
