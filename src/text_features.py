from __future__ import annotations
from collections import Counter
import re
import numpy as np


class BagOfWords:
    """
    Build a vocabulary and transform documents into count vectors.
    """

    def __init__(self, max_features: int = 5000) -> None:
        """Create a bag-of-words vectorizer."""
        if max_features <= 0:
            raise ValueError("max_features must be positive.")
        self.max_features = max_features
        self.vocabulary_: dict[str, int] | None = None

    @staticmethod
    def _tokenize(document: str) -> list[str]:
        """Lowercase and split a document into word tokens."""
        return re.findall(r"\b[a-zA-Z][a-zA-Z]+\b", document.lower())

    def fit(self, documents: list[str]) -> "BagOfWords":
        """Build a top-frequency vocabulary from training documents."""
        token_counts: Counter[str] = Counter()
        for document in documents:
            token_counts.update(self._tokenize(document))

        most_common = token_counts.most_common(self.max_features)
        self.vocabulary_ = {token: index for index, (token, _)
                            in enumerate(most_common)}
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        """Convert documents into a dense count matrix."""
        if self.vocabulary_ is None:
            raise ValueError("BagOfWords is not fitted yet.")

        counts = np.zeros((len(documents), len(self.vocabulary_)), dtype=float)
        for row_index, document in enumerate(documents):
            document_counts = Counter(self._tokenize(document))
            for token, count in document_counts.items():
                column_index = self.vocabulary_.get(token)
                if column_index is not None:
                    counts[row_index, column_index] = count
        return counts


class TfidfTransformer:
    """Convert bag-of-words count vectors to TF-IDF weights."""

    def __init__(self) -> None:
        """Create an unfitted TF-IDF transformer."""
        self.idf_: np.ndarray | None = None

    def fit(self, count_matrix: np.ndarray) -> "TfidfTransformer":
        """Estimate inverse document frequency values from counts."""
        count_matrix = np.asarray(count_matrix, dtype=float)
        if count_matrix.ndim != 2:
            raise ValueError("count_matrix must be a 2D array.")

        n_documents = count_matrix.shape[0]
        document_frequency = np.sum(count_matrix > 0, axis=0)
        self.idf_ = np.log(n_documents / (1.0 + document_frequency))
        return self

    def transform(self, count_matrix: np.ndarray) -> np.ndarray:
        """Transform count vectors into TF-IDF vectors."""
        if self.idf_ is None:
            raise ValueError("TfidfTransformer is not fitted yet.")

        count_matrix = np.asarray(count_matrix, dtype=float)
        if count_matrix.ndim != 2:
            raise ValueError("count_matrix must be a 2D array.")

        return count_matrix * self.idf_