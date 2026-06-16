from __future__ import annotations
import numpy as np

class RidgeRegression:
    """
    Ridge regression using the closed-form solution.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        """ Create a Ridge regression model. """
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative")
        self.lambda_ = lambda_
        self.weights_ : np.ndarray | None = None
        self.coef_ : np.ndarray | None = None
        self.intercept_ : float| None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """Fit ridge regression with a vectorized closed-form solution."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same row count.")

        X_aug = np.c_[np.ones(X.shape[0]), X]
        X_transpose_X = X_aug.T @ X_aug
        X_transpose_y = X_aug.T @ y

        penalty = np.eye(X_aug.shape[1])
        penalty[0,0] = 0

        try:
            self.weights_ = np.linalg.inv(X_transpose_X + self.lambda_ * penalty) @ X_transpose_y
        except np.linalg.LinAlgError:
            self.weights_ = np.linalg.pinv(X_transpose_X + self.lambda_ * penalty) @ X_transpose_y
        self.coef_ = self.weights_[1:].copy()
        self.intercept_ = float(self.weights_[0])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for new samples."""
        if self.weights_ is None:
            raise ValueError("Model must be fitted before prediction.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        X_aug = np.c_[np.ones(X.shape[0]), X]
        return X_aug @ self.weights_