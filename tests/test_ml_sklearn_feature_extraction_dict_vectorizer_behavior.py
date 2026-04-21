from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction import DictVectorizer


def _records() -> tuple[dict[str, int | float | str | tuple[str, ...]], ...]:
    return (
        {"city": "nyc", "temp": 71.5, "tags": ("coastal", "dense")},
        {"city": "sf", "temp": 64.0, "tags": ("coastal",), "rain": 0},
        {"city": "nyc", "temp": 68.0, "tags": ("dense", "dense")},
    )


def test_dict_vectorizer_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import (
        DictVectorizerState,
        dict_vectorizer_feature_names,
        dict_vectorizer_fit,
        dict_vectorizer_inverse_transform,
        dict_vectorizer_restrict,
        dict_vectorizer_transform,
    )

    assert DictVectorizerState is not None
    assert callable(dict_vectorizer_feature_names)
    assert callable(dict_vectorizer_fit)
    assert callable(dict_vectorizer_inverse_transform)
    assert callable(dict_vectorizer_restrict)
    assert callable(dict_vectorizer_transform)


def test_dict_vectorizer_fit_transform_matches_sklearn_sorted_dense() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import dict_vectorizer_fit, dict_vectorizer_transform

    records = _records()
    expected = DictVectorizer(sparse=False).fit(list(records))
    state = dict_vectorizer_fit(records)

    assert state.feature_names == tuple(expected.feature_names_)
    assert state.vocabulary == expected.vocabulary_
    assert np.array_equal(dict_vectorizer_transform(records, state), expected.transform(list(records)))


def test_dict_vectorizer_transform_ignores_unknown_features_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import dict_vectorizer_fit, dict_vectorizer_transform

    records = _records()
    query = (
        {"city": "nyc", "temp": 70.0, "unseen": 5.0, "tags": ("dense", "new")},
        {"city": "la", "temp": 80.0},
    )
    expected = DictVectorizer(sparse=False).fit(list(records))
    state = dict_vectorizer_fit(records)

    assert np.array_equal(dict_vectorizer_transform(query, state), expected.transform(list(query)))


def test_dict_vectorizer_unsorted_separator_and_feature_names_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import dict_vectorizer_feature_names, dict_vectorizer_fit, dict_vectorizer_transform

    records = _records()
    expected = DictVectorizer(sparse=False, sort=False, separator=":").fit(list(records))
    state = dict_vectorizer_fit(records, sort=False, separator=":")

    assert dict_vectorizer_feature_names(state) == tuple(expected.get_feature_names_out())
    assert np.array_equal(dict_vectorizer_transform(records, state), expected.transform(list(records)))


def test_dict_vectorizer_inverse_transform_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import (
        dict_vectorizer_fit,
        dict_vectorizer_inverse_transform,
        dict_vectorizer_transform,
    )

    records = _records()
    expected = DictVectorizer(sparse=False).fit(list(records))
    state = dict_vectorizer_fit(records)
    matrix = dict_vectorizer_transform(records, state)

    assert dict_vectorizer_inverse_transform(matrix, state) == expected.inverse_transform(matrix)


def test_dict_vectorizer_restrict_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import (
        dict_vectorizer_fit,
        dict_vectorizer_restrict,
        dict_vectorizer_transform,
    )

    records = _records()
    support = (0, 2, 4)
    expected = DictVectorizer(sparse=False).fit(list(records))
    expected.restrict(np.array(support, dtype=np.int64), indices=True)
    state = dict_vectorizer_restrict(dict_vectorizer_fit(records), support)

    assert state.feature_names == tuple(expected.feature_names_)
    assert state.vocabulary == expected.vocabulary_
    assert np.array_equal(dict_vectorizer_transform(records, state), expected.transform(list(records)))


def test_dict_vectorizer_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import dict_vectorizer_fit, dict_vectorizer_restrict, dict_vectorizer_transform

    state = dict_vectorizer_fit(_records())
    with pytest.raises(Exception):
        dict_vectorizer_fit(())
    with pytest.raises(Exception):
        dict_vectorizer_fit(({"nested": {"bad": 1}},))
    with pytest.raises(Exception):
        dict_vectorizer_transform((), state)
    with pytest.raises(Exception):
        dict_vectorizer_restrict(state, (len(state.feature_names),))
