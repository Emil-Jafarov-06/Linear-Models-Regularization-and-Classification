from __future__ import annotations
import numpy as np

class LogisticRegression:
    """
    Logistic regression with batch gradient descent.
    """

    def __init__(self, lr: float = 0.1, lambda_: float = 0.0, max_iter: int = 1000) -> None:
        """Create a logistic regression model."""
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        self.lr = lr
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.classes_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Compute the sigmoid function in a numerically stable way."""
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z, dtype=float)
        positive = z >= 0

        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

        exp_z = np.exp(z[~positive])
        out[~positive] = exp_z / (1.0 + exp_z)

        return out

    def _fit_binary(self, X_aug: np.ndarray, y_binary: np.ndarray) -> np.ndarray:
        """Train one binary logistic-regression classifier."""
        n_samples, n_features = X_aug.shape[0], X_aug.shape[1]
        weights = np.zeros(n_features, dtype=float)

        for _ in range(self.max_iter):
            probabilities = self._sigmoid(X_aug @ weights)
            gradient = (X_aug.T @ (probabilities - y_binary)) / n_samples
            regularization = (self.lambda_ / n_samples) * weights
            regularization[0] = 0.0
            weights -= self.lr * (gradient + regularization)

        return weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit binary logistic regression or multiclass OvR logistic regression."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same row count.")

        self.classes_ = np.unique(y)
        X_aug = np.c_[np.ones(X.shape[0]), X]

        if len(self.classes_) == 2:
            y_binary = (y == self.classes_[1]).astype(float)
            self.weights_ = self._fit_binary(X_aug, y_binary)
        else:
            all_weights = []
            for class_label in self.classes_:
                y_binary = (y == class_label).astype(float)
                all_weights.append(self._fit_binary(X_aug, y_binary))
            self.weights_ = np.vstack(all_weights)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each sample."""
        if self.weights_ is None or self.classes_ is None:
            raise ValueError("Model is not fitted yet.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        X_aug = np.c_[np.ones(X.shape[0]), X]
        if len(self.classes_) == 2:
            positive_probability = self._sigmoid(X_aug @ self.weights_)
            return np.c_[1.0 - positive_probability, positive_probability]

        scores = self._sigmoid(X_aug @ self.weights_.T)
        row_sums = scores.sum(axis=1, keepdims=True)
        return scores / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for new samples."""
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]
