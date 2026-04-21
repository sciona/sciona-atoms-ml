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
