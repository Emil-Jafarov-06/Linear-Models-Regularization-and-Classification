from __future__ import annotations
import numpy as np

class WeightedLinearRegression:
    """
    Weighted least-squares linear regression.
    """

    def __init__(self) -> None:
        """Create an unfitted weighted linear-regression model."""
        self.weights_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "WeightedLinearRegression":
        """Fit weighted least squares to ``X`` and ``y``.

        Parameters:
        X: Training feature matrix with shape ``(n_samples, n_features)``.
        y: Target vector with shape ``(n_samples,)``.
        sample_weight:Positive sample weights with shape ``(n_samples,)``.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if sample_weight.ndim != 1:
            raise ValueError("sample_weight must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same row count.")
        if sample_weight.shape[0] != X.shape[0]:
            raise ValueError("sample_weight must have one value per sample.")
        if np.any(sample_weight <= 0):
            raise ValueError("All sample weights must be positive.")

        X_aug = np.c_[np.ones(X.shape[0]), X]
        weighted_X = X_aug * sample_weight[:, np.newaxis]
        normal_matrix = X_aug.T @ weighted_X
        normal_rhs = X_aug.T @ (sample_weight * y)

        try:
            self.weights_ = np.linalg.solve(normal_matrix, normal_rhs)
        except np.linalg.LinAlgError:
            self.weights_ = np.linalg.pinv(normal_matrix) @ normal_rhs

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
