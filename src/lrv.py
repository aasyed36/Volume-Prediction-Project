import numpy as np
from sklearn.linear_model import LinearRegression


class LinearRegressionVolume:
    """Modeling trading volumes using linear regression."""

    def __init__(self, dim, **lr_args):
        self.dim = dim
        self.model = [LinearRegression(**lr_args) for _ in range(dim)]

    def fit(self, X):
        """Fit the linear regression model."""
        self.n_assets = X.shape[1] // self.dim
        for i in range(self.dim):
            if i == 0:
                self.model[i].fit(np.zeros((X.shape[0], 1)), X[:, self.n_assets * i:])
            else:
                self.model[i].fit(X[:, :self.n_assets * i], X[:, self.n_assets * i:])

    def predict(self, X=None):
        """Predict the trading volumes."""
        if X is None:
            return self.model[0].predict(np.zeros((1, 1)))

        return self.model[X.shape[1] // self.n_assets].predict(X)
