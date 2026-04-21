from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer


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


def test_hashing_vectorizer_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import hashing_vectorizer_token, hashing_vectorizer_transform

    assert callable(hashing_vectorizer_token)
    assert callable(hashing_vectorizer_transform)


@pytest.mark.parametrize("alternate_sign", [True, False])
def test_hashing_vectorizer_token_matches_feature_hasher(alternate_sign: bool) -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import hashing_vectorizer_token

    token = "document"
    column, sign = hashing_vectorizer_token(token, n_features=100, alternate_sign=alternate_sign)
    expected = FeatureHasher(n_features=100, input_type="string", alternate_sign=alternate_sign).transform([[token]])

    assert expected.indices.tolist() == [column]
    assert expected.data.tolist() == [sign]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_features": 64, "norm": None},
        {"n_features": 64, "alternate_sign": False, "norm": None},
        {"n_features": 64, "ngram_range": (1, 2), "stop_words": ("is", "the"), "norm": "l2"},
        {"n_features": 32, "binary": True, "norm": None},
        {"n_features": 32, "binary": True, "norm": "l1", "alternate_sign": True},
        {"n_features": 64, "lowercase": False, "norm": None},
        {"n_features": 64, "strip_accents": "unicode", "token_pattern": r"(?u)\b\w+\b", "norm": "l2"},
    ],
)
def test_hashing_vectorizer_transform_matches_sklearn(kwargs: dict[str, object]) -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import hashing_vectorizer_transform

    corpus = _corpus()
    matrix = hashing_vectorizer_transform(corpus, **kwargs)
    expected = HashingVectorizer(**_sklearn_kwargs(kwargs)).transform(corpus).toarray().astype(np.float64)

    assert np.allclose(matrix, expected)


def test_hashing_vectorizer_binary_preserves_signed_collision_presence() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import hashing_vectorizer_transform

    corpus = ("document cat",)
    matrix = hashing_vectorizer_transform(corpus, n_features=1, binary=True, norm=None, alternate_sign=True)
    expected = HashingVectorizer(n_features=1, binary=True, norm=None, alternate_sign=True).transform(corpus).toarray()

    assert np.array_equal(matrix, expected)
    assert matrix[0, 0] == 1.0


def test_hashing_vectorizer_rejects_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import hashing_vectorizer_token, hashing_vectorizer_transform

    corpus = _corpus()

    with pytest.raises(Exception):
        hashing_vectorizer_token("", n_features=8)
    with pytest.raises(Exception):
        hashing_vectorizer_token("document", n_features=0)
    with pytest.raises(Exception):
        hashing_vectorizer_transform(())
    with pytest.raises(Exception):
        hashing_vectorizer_transform(corpus, ngram_range=(2, 1))
    with pytest.raises(Exception):
        hashing_vectorizer_transform(corpus, token_pattern=r"(\w)(\w)")
    with pytest.raises(Exception):
        hashing_vectorizer_transform(corpus, n_features=0)
    with pytest.raises(Exception):
        hashing_vectorizer_transform(corpus, norm="max")
