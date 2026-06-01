from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tokenize_and_count_words,
    witness_apply_tfidf_weighting,
)

@register_atom(witness_tokenize_and_count_words, name="tokenize_and_count_words")
@icontract.require(lambda corpus, ngram_range, min_df: len(corpus) > 0, "Precondition failed: len(corpus) > 0")
@icontract.ensure(lambda result, corpus, ngram_range, min_df: len(vocabulary) > 0, "Postcondition failed: len(vocabulary) > 0")
def tokenize_and_count_words(corpus: str, ngram_range: int, min_df: float) -> Any:
    """Construct occurrences dictionary and count frequencies across documents.

    Args:
        corpus: list[str]
        ngram_range: tuple[int, int]
        min_df: float

    Returns:
        raw_counts: Any
    """
    import sklearn.feature_extraction.text
    return sklearn.feature_extraction.text.CountVectorizer(corpus=corpus, ngram_range=ngram_range, min_df=min_df) # type: ignore

@register_atom(witness_apply_tfidf_weighting, name="apply_tfidf_weighting")
@icontract.require(lambda raw_counts, norm: raw_counts is not None, "Precondition failed: raw_counts is not None")
@icontract.ensure(lambda result, raw_counts, norm: result is not None, "Postcondition failed: result is not None")
def apply_tfidf_weighting(raw_counts: Any, norm: str) -> Any:
    """Transform raw occurrences into relative importance weights using IDF logarithmic scaling.

    Args:
        raw_counts: Any
        norm: str

    Returns:
        tfidf_matrix: Any
    """
    import sklearn.feature_extraction.text
    return sklearn.feature_extraction.text.TfidfTransformer(raw_counts=raw_counts, norm=norm) # type: ignore

