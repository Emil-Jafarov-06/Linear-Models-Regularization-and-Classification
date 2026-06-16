from __future__ import annotations
import numpy as np

class LassoRegression:
    """
    Lasso regression trained with cyclic coordinate descent.
    """

    def __init__(self, lambda_: float = 1.0, max_iter: int = 10000, tol: float = 1e-4) -> None:
        """Create a lasso regression model."""
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")

        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.weights_standardized_: np.ndarray | None = None
        self.n_iter_: int = 0
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.y_mean_: float | None = None

    @staticmethod
    def _soft_threshold(value: float, penalty: float) -> float:
        """Apply the soft-thresholding operator."""
        if value > penalty:
            return value - penalty
        if value < -penalty:
            return value + penalty
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegression":
        """Fit lasso regression with coordinate descent."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same row count.")

        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ == 0.0] = 1.0
        self.y_mean_ = float(y.mean())

        X_std = (X - self.x_mean_) / self.x_scale_
        y_centered = y - self.y_mean_
        weights = np.zeros(X.shape[1], dtype=float)

        for iteration in range(0, self.max_iter):
            old_weights = weights.copy()

            for j in range(X.shape[1]):
                residual_without_j = y_centered - X_std @ weights + X_std[:, j] * weights[j]
                rho_j = float(X_std[:, j] @ residual_without_j)
                denominator_j = float(X_std[:, j] @ X_std[:, j])
                weights[j] = self._soft_threshold(rho_j, self.lambda_) / denominator_j

            max_change = float(np.max(np.abs(weights - old_weights)))
            if max_change < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.weights_standardized_ = weights.copy()
        self.coef_ = weights / self.x_scale_
        self.intercept_ = self.y_mean_ - float(self.x_mean_ @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for new samples."""
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model is not fitted yet.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        return self.intercept_ + X @ self.coef_
