"""Ghost witnesses for deterministic text and NLP helper atoms."""

from __future__ import annotations

from collections.abc import Callable

from numpy.typing import NDArray

from sciona.ghost.abstract import AbstractArray


def witness_clean_text(text: str, operations: list[str], vocab_freq: dict[str, int] | None = None) -> str:
    """Describe a text normalization result."""
    del vocab_freq
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if not operations:
        raise ValueError("operations must not be empty")
    return text


def witness_levenshtein(s1: str, s2: str) -> int:
    """Describe an edit-distance scalar."""
    del s1, s2
    return 0


def witness_jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """Describe a bounded string-similarity score."""
    del s1, s2
    if prefix_weight < 0.0 or prefix_weight > 0.25:
        raise ValueError("prefix_weight must be in [0, 0.25]")
    return 1.0


def witness_bio_decode(tags: list[str], tokens: list[str]) -> AbstractArray:
    """Describe decoded sequence-label spans."""
    if len(tags) != len(tokens):
        raise ValueError("tags and tokens must have equal length")
    return AbstractArray(shape=(len(tags), 3), dtype="object")


def witness_char_to_token_offsets(char_spans: list[tuple[int, int]], offset_mapping: list[tuple[int, int]]) -> AbstractArray:
    """Describe character-span to token-span alignment."""
    del offset_mapping
    return AbstractArray(shape=(len(char_spans), 2), dtype="int64", min_val=0.0)


def witness_beam_search(
    log_probs_fn: Callable[[tuple[tuple[int, ...], ...]], NDArray[object]],
    start_token: int,
    end_token: int,
    beam_width: int,
    max_length: int,
    alpha: float = 0.7,
) -> AbstractArray:
    """Describe a bounded list of decoded token sequences."""
    del log_probs_fn, start_token, end_token, alpha
    if beam_width < 1 or max_length < 1:
        raise ValueError("beam_width and max_length must be positive")
    return AbstractArray(shape=(beam_width, max_length), dtype="object")


def witness_feature_hash(tokens: list[str], n_features: int) -> AbstractArray:
    """Describe sparse hashed token counts."""
    del tokens
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0)


def witness_word_ngrams(tokens: list[str], n: int) -> AbstractArray:
    """Describe rolling token n-grams."""
    if n < 1:
        raise ValueError("n must be positive")
    return AbstractArray(shape=(max(0, len(tokens) - n + 1), n), dtype="object")


def witness_char_ngrams(text: str, n: int) -> AbstractArray:
    """Describe rolling character n-grams."""
    if n < 1:
        raise ValueError("n must be positive")
    return AbstractArray(shape=(max(0, len(text) - n + 1),), dtype="object")


def witness_filter_spans_by_length(spans: list[tuple[str, int, int]], min_lengths: dict[str, int]) -> AbstractArray:
    """Describe class-thresholded span output."""
    del min_lengths
    return AbstractArray(shape=(len(spans), 3), dtype="object")
