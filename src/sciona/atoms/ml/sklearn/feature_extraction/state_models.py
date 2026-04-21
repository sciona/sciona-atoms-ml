"""State containers for sklearn feature extraction atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DictVectorizerState:
    """Learned feature names and vocabulary for dictionary vectorization."""

    feature_names: tuple[str, ...]
    vocabulary: dict[str, int]
    separator: str


@dataclass(frozen=True)
class TfidfTransformerState:
    """Learned inverse-document-frequency weights for dense TF-IDF transforms."""

    idf: NDArray[np.float64] | None
    norm: str | None
    use_idf: bool
    smooth_idf: bool
    sublinear_tf: bool
    n_features_in: int


@dataclass(frozen=True)
class CountVectorizerState:
    """Learned dense count-vectorizer vocabulary and text-analysis settings."""

    vocabulary: dict[str, int]
    feature_names: tuple[str, ...]
    lowercase: bool
    strip_accents: str | None
    token_pattern: str
    ngram_range: tuple[int, int]
    stop_words: tuple[str, ...] | None
    binary: bool
    fixed_vocabulary: bool


@dataclass(frozen=True)
class TfidfVectorizerState:
    """Learned count vocabulary and IDF weights for dense TF-IDF vectorization."""

    count_state: CountVectorizerState
    tfidf_state: TfidfTransformerState
