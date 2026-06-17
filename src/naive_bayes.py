from __future__ import annotations
import numpy as np

class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier using log-space prediction."""

    def __init__(self) -> None:
        """Create an unfitted Gaussian Naive Bayes model."""
        self.classes_: np.ndarray | None = None
        self.class_priors_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None
        self.epsilon_: float = 1e-9

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        """Estimate class priors, class means, and class variances."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same row count.")

        self.classes_ = np.unique(y)
        means = []
        variances = []
        priors = []

        for class_label in self.classes_:
            X_class = X[y == class_label]
            means.append(X_class.mean(axis=0))
            variances.append(X_class.var(axis=0) + self.epsilon_)
            priors.append(X_class.shape[0] / X.shape[0])

        self.means_ = np.vstack(means)
        self.variances_ = np.vstack(variances)
        self.class_priors_ = np.asarray(priors, dtype=float)
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log p(x, y=c) for every sample and class."""
        if self.classes_ is None or self.means_ is None or self.variances_ is None or self.class_priors_ is None:
            raise ValueError("Model is not fitted yet.")

        X = np.asarray(X, dtype=float)
        log_priors = np.log(self.class_priors_)
        log_likelihoods = []

        for class_index in range(len(self.classes_)):
            mean = self.means_[class_index]
            variance = self.variances_[class_index]
            log_density = -0.5 * np.sum(
                np.log(2.0 * np.pi * variance) + ((X - mean) ** 2) / variance,
                axis=1,
            )
            log_likelihoods.append(log_priors[class_index] + log_density)

        return np.vstack(log_likelihoods).T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalized posterior probabilities."""
        log_joint = self._joint_log_likelihood(X)
        max_log = np.max(log_joint, axis=1, keepdims=True)
        exp_shifted = np.exp(log_joint - max_log)
        return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for new samples."""
        if self.classes_ is None:
            raise ValueError("Model is not fitted yet.")
        log_joint = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(log_joint, axis=1)]
