from __future__ import annotations
import numpy as np

class LinearRegressionGD:
    """
    Linear regression trained by batch gradient descent on MSE.
    """

    def __init__(self, lr: float = 0.01, max_iter: int = 1000, tol: float = 1e-6) -> None:
        """Create a gradient-descent linear regression model."""
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")

        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.weights_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None
        self.mse_history_: list[float] = []
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        """Fit the model using batch gradient descent."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("y must be a 1D array with one target per row of X.")

        n_samples = X.shape[0]
        X_aug = np.c_[np.ones(n_samples), X]
        weights = np.zeros(X_aug.shape[1], dtype=float)
        self.mse_history_ = list()

        for iteration in range(self.max_iter):
            predictions = X_aug @ weights
            errors = predictions - y
            mse = float(np.mean(errors ** 2))
            self.mse_history_.append(mse)

            new_weights = weights - self.lr * (2.0 / n_samples) * (X_aug.T @ errors)

            if np.linalg.norm(new_weights - weights, ord=2) < self.tol:
                weights = new_weights
                self.n_iter_ = iteration + 1
                break

            weights = new_weights
        else:
            self.n_iter_ = self.max_iter

        self.weights_ = weights
        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:].copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for new samples."""
        if self.weights_ is None:
            raise ValueError("Model is not fitted yet.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        X_aug = np.c_[np.ones(X.shape[0]), X]
        return X_aug @ self.weights_
