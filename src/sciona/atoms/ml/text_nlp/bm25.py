"""Reusable count-based BM25 scoring without text or corpus-fitting policy."""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete aligned counts and numerical controls required')


@register_atom(concrete_only)
def score_bm25(term_frequencies: list[int], document_frequencies: list[int],
               document_length: int, document_count: int, average_length: float,
               k1: float, b: float) -> float:
    """Score aligned distinct query terms against one document using BM25.

    Uses log1p((N-df+.5)/(df+.5)) IDF and tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl)).
    Query terms have unit weight. Counts are Python integers in [0,2**53];
    corpus size is positive, df <= N, and sum(tf) <= document length.
    Average length and k1 are positive finite values <= 2**53; b is in [0,1].
    Empty queries and zero-length documents score zero. Unseen terms may have
    df zero. Rejects nonfinite intermediates/results; inputs are unchanged.

    Caller supplies distinct-term alignment, corpus statistics and fitting
    populations. No tokenization, text normalization, fitting, labels, query
    weighting or task-specific policy. Positive IDF, exact lengths and (k1+1)
    scale are explicit; no bitwise Lucene-equivalence claim.
    """
    from sciona.bm25_scoring import bm25_score
    return bm25_score(term_frequencies, document_frequencies, document_length,
                      document_count, average_length, k1, b)
