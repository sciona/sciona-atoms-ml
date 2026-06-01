from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_tokenize_and_count_words(corpus: AbstractScalar | str, ngram_range: AbstractScalar | int, min_df: AbstractScalar | float) -> AbstractScalar:
    """Ghost witness for tokenize_and_count_words."""
    _ = (corpus, ngram_range, min_df)
    return AbstractScalar(dtype="float64")

def witness_apply_tfidf_weighting(raw_counts: AbstractScalar | Any, norm: AbstractScalar | str) -> AbstractScalar:
    """Ghost witness for apply_tfidf_weighting."""
    _ = (raw_counts, norm)
    return AbstractScalar(dtype="float64")

