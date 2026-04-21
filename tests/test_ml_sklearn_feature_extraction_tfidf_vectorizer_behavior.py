from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer


def _corpus() -> tuple[str, ...]:
    return (
        "This is the first document.",
        "This document is the second document.",
        "Cafe document and cafe tokens.",
        "Is this the first document?",
    )


def _sklearn_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    converted = dict(kwargs)
    if isinstance(converted.get("stop_words"), tuple):
        converted["stop_words"] = list(converted["stop_words"])
    return converted


def test_tfidf_vectorizer_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import (
        TfidfVectorizerState,
        tfidf_vectorizer_feature_names,
        tfidf_vectorizer_fit,
        tfidf_vectorizer_idf,
        tfidf_vectorizer_transform,
    )

    assert TfidfVectorizerState is not None
    assert callable(tfidf_vectorizer_fit)
    assert callable(tfidf_vectorizer_transform)
    assert callable(tfidf_vectorizer_feature_names)
    assert callable(tfidf_vectorizer_idf)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"norm": None},
        {"use_idf": False, "norm": None},
        {"smooth_idf": False, "sublinear_tf": True},
        {"binary": True, "use_idf": False, "norm": None},
        {"ngram_range": (1, 2), "stop_words": ("is", "the")},
        {"max_df": 0.75, "min_df": 1},
        {"max_features": 4},
        {"vocabulary": ("this", "document", "missing")},
        {"lowercase": False},
        {"strip_accents": "unicode", "token_pattern": r"(?u)\b\w+\b"},
    ],
)
def test_tfidf_vectorizer_fit_transform_matches_sklearn(kwargs: dict[str, object]) -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import (
        tfidf_vectorizer_feature_names,
        tfidf_vectorizer_fit,
        tfidf_vectorizer_transform,
    )

    corpus = _corpus()
    state = tfidf_vectorizer_fit(corpus, **kwargs)
    matrix = tfidf_vectorizer_transform(corpus, state)
    expected = TfidfVectorizer(**_sklearn_kwargs(kwargs)).fit(corpus)

    assert tfidf_vectorizer_feature_names(state) == tuple(expected.get_feature_names_out())
    assert np.allclose(matrix, expected.transform(corpus).toarray())


def test_tfidf_vectorizer_idf_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import tfidf_vectorizer_fit, tfidf_vectorizer_idf

    corpus = _corpus()
    state = tfidf_vectorizer_fit(corpus, smooth_idf=False)
    expected = TfidfVectorizer(smooth_idf=False).fit(corpus)

    assert np.allclose(tfidf_vectorizer_idf(state), expected.idf_)


def test_tfidf_vectorizer_transform_ignores_out_of_vocabulary_terms() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import tfidf_vectorizer_fit, tfidf_vectorizer_transform

    corpus = _corpus()
    query = ("unknown document token",)
    state = tfidf_vectorizer_fit(corpus)
    expected = TfidfVectorizer().fit(corpus)

    assert np.allclose(tfidf_vectorizer_transform(query, state), expected.transform(query).toarray())


def test_tfidf_vectorizer_rejects_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import tfidf_vectorizer_fit, tfidf_vectorizer_idf, tfidf_vectorizer_transform

    corpus = _corpus()
    state = tfidf_vectorizer_fit(corpus)
    no_idf_state = tfidf_vectorizer_fit(corpus, use_idf=False)

    with pytest.raises(Exception):
        tfidf_vectorizer_fit(())
    with pytest.raises(Exception):
        tfidf_vectorizer_fit(corpus, ngram_range=(2, 1))
    with pytest.raises(Exception):
        tfidf_vectorizer_fit(corpus, token_pattern=r"(\w)(\w)")
    with pytest.raises(Exception):
        tfidf_vectorizer_fit(corpus, vocabulary=("dup", "dup"))
    with pytest.raises(Exception):
        tfidf_vectorizer_fit(corpus, norm="max")
    with pytest.raises(Exception):
        tfidf_vectorizer_transform((), state)
    with pytest.raises(Exception):
        tfidf_vectorizer_idf(no_idf_state)
