from __future__ import annotations
import numpy as np

class LinearRegression:
    """
    Linear regression using the closed-form normal equation.
    """

    def __init__(self) -> None:
        """Create an unfitted linear regression model."""
        self.weights_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit the model to training data.

        X: Training feature matrix with shape ``(n_samples, n_features)``.
        y: Target vector with shape ``(n_samples,)``.

        Returns the fitted model of LinearRegression.
        """
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
        X_transpose_Y = X_aug.T @ y

        try:
            self.weights_ = np.linalg.inv(X_transpose_X) @ X_transpose_Y
        except np.linalg.LinAlgError:
            self.weights_ = np.linalg.pinv(X_transpose_X) @ X_transpose_Y

        self.intercept_ = float(self.weights_[0])
        self.coef_ = self.weights_[1:].copy()
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
