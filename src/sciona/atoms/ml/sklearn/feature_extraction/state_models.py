"""State containers for sklearn feature extraction atoms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DictVectorizerState:
    """Learned feature names and vocabulary for dictionary vectorization."""

    feature_names: tuple[str, ...]
    vocabulary: dict[str, int]
    separator: str
