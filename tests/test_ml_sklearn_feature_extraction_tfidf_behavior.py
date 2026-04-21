from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfTransformer


def _count_matrix() -> np.ndarray:
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_tfidf_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import (
        TfidfTransformerState,
        tfidf_document_frequency,
        tfidf_idf,
        tfidf_transform,
        tfidf_transformer_fit,
    )

    assert TfidfTransformerState is not None
    assert callable(tfidf_document_frequency)
    assert callable(tfidf_idf)
    assert callable(tfidf_transformer_fit)
    assert callable(tfidf_transform)


def test_tfidf_document_frequency_and_idf_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import tfidf_document_frequency, tfidf_idf

    X = _count_matrix()
    expected = TfidfTransformer(norm=None, smooth_idf=True).fit(X)
    document_frequency = tfidf_document_frequency(X)
    idf = tfidf_idf(document_frequency, X.shape[0], smooth_idf=True)

    assert np.array_equal(document_frequency, np.array([4.0, 3.0, 2.0, 4.0, 1.0, 4.0, 1.0, 1.0]))
    assert np.allclose(idf, expected.idf_)


def test_tfidf_fit_state_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import tfidf_transformer_fit

    X = _count_matrix()
    expected = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=False, sublinear_tf=True).fit(X)
    state = tfidf_transformer_fit(X, norm="l2", use_idf=True, smooth_idf=False, sublinear_tf=True)

    assert state.n_features_in == X.shape[1]
    assert state.norm == "l2"
    assert state.use_idf is True
    assert state.smooth_idf is False
    assert state.sublinear_tf is True
    assert state.idf is not None
    assert np.allclose(state.idf, expected.idf_)


@pytest.mark.parametrize("norm", ["l2", "l1", None])
@pytest.mark.parametrize("use_idf", [True, False])
@pytest.mark.parametrize("smooth_idf", [True, False])
@pytest.mark.parametrize("sublinear_tf", [True, False])
def test_tfidf_transform_matches_sklearn(
    norm: str | None,
    use_idf: bool,
    smooth_idf: bool,
    sublinear_tf: bool,
) -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import tfidf_transform, tfidf_transformer_fit

    X = _count_matrix()
    expected = TfidfTransformer(
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    ).fit(X)
    state = tfidf_transformer_fit(
        X,
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    )

    if use_idf:
        assert state.idf is not None
        assert np.allclose(state.idf, expected.idf_)
    else:
        assert state.idf is None
    assert np.allclose(tfidf_transform(X, state), expected.transform(X).toarray())


def test_tfidf_rejects_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import tfidf_idf, tfidf_transform, tfidf_transformer_fit

    X = _count_matrix()
    state = tfidf_transformer_fit(X)

    with pytest.raises(Exception):
        tfidf_transformer_fit(-X)
    with pytest.raises(Exception):
        tfidf_transformer_fit(np.ones((0, 3), dtype=np.float64))
    with pytest.raises(Exception):
        tfidf_transformer_fit(X, norm="max")
    with pytest.raises(Exception):
        tfidf_idf(np.array([5.0], dtype=np.float64), 4)
    with pytest.raises(Exception):
        tfidf_transform(np.ones((2, 2), dtype=np.float64), state)
