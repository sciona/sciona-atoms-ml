from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer


def _corpus() -> tuple[str, ...]:
    return (
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    )


def _sklearn_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    converted = dict(kwargs)
    if isinstance(converted.get("stop_words"), tuple):
        converted["stop_words"] = list(converted["stop_words"])
    return converted


def test_count_vectorizer_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import (
        CountVectorizerState,
        count_vectorizer_analyze,
        count_vectorizer_feature_names,
        count_vectorizer_fit,
        count_vectorizer_inverse_transform,
        count_vectorizer_transform,
    )

    assert CountVectorizerState is not None
    assert callable(count_vectorizer_analyze)
    assert callable(count_vectorizer_fit)
    assert callable(count_vectorizer_transform)
    assert callable(count_vectorizer_feature_names)
    assert callable(count_vectorizer_inverse_transform)


def test_count_vectorizer_analyze_matches_sklearn_default_and_bigrams() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import count_vectorizer_analyze

    document = "This is the first document."
    expected_default = CountVectorizer().build_analyzer()(document)
    expected_bigrams = CountVectorizer(ngram_range=(1, 2), stop_words=["is", "the"]).build_analyzer()(document)

    assert count_vectorizer_analyze(document) == tuple(expected_default)
    assert count_vectorizer_analyze(document, ngram_range=(1, 2), stop_words=("is", "the")) == tuple(expected_bigrams)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"ngram_range": (1, 2)},
        {"stop_words": ("is", "the"), "ngram_range": (1, 2)},
        {"binary": True},
        {"max_df": 0.75, "min_df": 1},
        {"max_features": 4},
        {"vocabulary": ("this", "document", "missing")},
        {"lowercase": False},
        {"strip_accents": "unicode", "token_pattern": r"(?u)\b\w+\b"},
    ],
)
def test_count_vectorizer_fit_transform_matches_sklearn(kwargs: dict[str, object]) -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import (
        count_vectorizer_feature_names,
        count_vectorizer_fit,
        count_vectorizer_inverse_transform,
        count_vectorizer_transform,
    )

    corpus = _corpus()
    state = count_vectorizer_fit(corpus, **kwargs)
    expected = CountVectorizer(**_sklearn_kwargs(kwargs)).fit(corpus)
    matrix = count_vectorizer_transform(corpus, state)
    target = expected.transform(corpus).toarray().astype(np.float64)

    assert count_vectorizer_feature_names(state) == tuple(expected.get_feature_names_out())
    assert np.array_equal(matrix, target)
    assert count_vectorizer_inverse_transform(matrix, state) == [tuple(row) for row in expected.inverse_transform(target)]


def test_count_vectorizer_transform_ignores_out_of_vocabulary_terms() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import count_vectorizer_fit, count_vectorizer_transform

    corpus = _corpus()
    query = ("unknown document token",)
    state = count_vectorizer_fit(corpus)
    expected = CountVectorizer().fit(corpus)

    assert np.array_equal(count_vectorizer_transform(query, state), expected.transform(query).toarray())


def test_count_vectorizer_rejects_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import count_vectorizer_fit, count_vectorizer_transform

    corpus = _corpus()
    state = count_vectorizer_fit(corpus)

    with pytest.raises(Exception):
        count_vectorizer_fit(())
    with pytest.raises(Exception):
        count_vectorizer_fit(corpus, ngram_range=(2, 1))
    with pytest.raises(Exception):
        count_vectorizer_fit(corpus, token_pattern=r"(\w)(\w)")
    with pytest.raises(Exception):
        count_vectorizer_fit(corpus, vocabulary=("dup", "dup"))
    with pytest.raises(Exception):
        count_vectorizer_fit(corpus, max_df=0.25, min_df=0.75)
    with pytest.raises(Exception):
        count_vectorizer_transform((), state)
