from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.preprocessing import Binarizer, Normalizer
from sklearn.preprocessing import add_dummy_feature as sklearn_add_dummy_feature
from sklearn.preprocessing import binarize as sklearn_binarize
from sklearn.preprocessing import normalize as sklearn_normalize
from sklearn.preprocessing import scale as sklearn_scale


def test_preprocessing_basic_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        add_dummy_feature,
        binarize,
        binarizer_transform,
        normalize,
        normalizer_transform,
        scale,
    )

    assert callable(add_dummy_feature)
    assert callable(binarize)
    assert callable(binarizer_transform)
    assert callable(normalize)
    assert callable(normalizer_transform)
    assert callable(scale)


def test_add_dummy_feature_matches_sklearn_dense() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import add_dummy_feature

    X = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    assert np.array_equal(add_dummy_feature(X, value=2.5), sklearn_add_dummy_feature(X, value=2.5))


def test_add_dummy_feature_matches_sklearn_sparse_formats() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import add_dummy_feature

    X_dense = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=np.float64)
    for matrix_factory in (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix):
        X = matrix_factory(X_dense)
        result = add_dummy_feature(X, value=-1.0)
        expected = sklearn_add_dummy_feature(X, value=-1.0)
        assert result.getformat() == expected.getformat()
        assert np.array_equal(result.toarray(), expected.toarray())


def test_binarize_matches_sklearn_dense_and_sparse() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import binarize

    X = np.array([[0.2, 0.6, 1.2], [0.5, -1.0, 0.0]], dtype=np.float64)
    assert np.array_equal(binarize(X, threshold=0.5), sklearn_binarize(X, threshold=0.5))

    X_sparse = sp.csr_matrix([[0.0, 0.7, 0.2], [1.4, 0.0, 0.3]], dtype=np.float64)
    result = binarize(X_sparse, threshold=0.25)
    expected = sklearn_binarize(X_sparse, threshold=0.25)
    assert np.array_equal(result.toarray(), expected.toarray())


def test_binarize_sparse_negative_threshold_error_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import binarize

    X = sp.csr_matrix([[0.0, 1.0]])
    with pytest.raises(ValueError, match="threshold < 0"):
        binarize(X, threshold=-0.1)
    with pytest.raises(ValueError, match="threshold < 0"):
        sklearn_binarize(X, threshold=-0.1)


def test_binarizer_transform_matches_sklearn_transform() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import binarizer_transform

    X = np.array([[1.0, -1.0, 2.0], [0.2, 0.4, 0.6]], dtype=np.float64)
    transformer = Binarizer(threshold=0.3).fit(X)
    assert np.array_equal(binarizer_transform(X, threshold=0.3), transformer.transform(X))

    X_sparse = sp.csr_matrix([[0.0, 0.7, 0.2], [1.4, 0.0, 0.3]], dtype=np.float64)
    sparse_result = binarizer_transform(X_sparse, threshold=0.25)
    sparse_expected = Binarizer(threshold=0.25).fit(X_sparse).transform(X_sparse)
    assert np.array_equal(sparse_result.toarray(), sparse_expected.toarray())


def test_normalize_matches_sklearn_dense_norms_and_axes() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import normalize

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    for norm in ("l1", "l2", "max"):
        assert np.allclose(normalize(X, norm=norm, axis=1), sklearn_normalize(X, norm=norm, axis=1))
        assert np.allclose(normalize(X, norm=norm, axis=0), sklearn_normalize(X, norm=norm, axis=0))

    result, norms = normalize(X, norm="l2", axis=1, return_norm=True)
    expected, expected_norms = sklearn_normalize(X, norm="l2", axis=1, return_norm=True)
    assert np.allclose(result, expected)
    assert np.allclose(norms, expected_norms)


def test_normalize_matches_sklearn_sparse_l1_l2_and_max_norms() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import normalize

    X = sp.csr_matrix([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    for norm in ("l1", "l2", "max"):
        result = normalize(X, norm=norm, axis=1)
        expected = sklearn_normalize(X, norm=norm, axis=1)
        assert np.allclose(result.toarray(), expected.toarray())

    max_result, max_norms = normalize(X, norm="max", axis=1, return_norm=True)
    expected_result, expected_norms = sklearn_normalize(X, norm="max", axis=1, return_norm=True)
    assert np.allclose(max_result.toarray(), expected_result.toarray())
    assert np.allclose(max_norms, expected_norms)


def test_normalize_sparse_return_norm_error_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import normalize

    X = sp.csr_matrix([[1.0, 2.0]])
    with pytest.raises(ValueError, match="return_norm=True"):
        normalize(X, norm="l2", return_norm=True)
    with pytest.raises(NotImplementedError, match="return_norm=True"):
        sklearn_normalize(X, norm="l2", return_norm=True)


def test_normalizer_transform_matches_sklearn_transform() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import normalizer_transform

    X = np.array([[4.0, 1.0, 2.0, 2.0], [1.0, 3.0, 9.0, 3.0]], dtype=np.float64)
    for norm in ("l1", "l2", "max"):
        transformer = Normalizer(norm=norm).fit(X)
        assert np.allclose(normalizer_transform(X, norm=norm), transformer.transform(X))

    X_sparse = sp.csr_matrix(X)
    sparse_result = normalizer_transform(X_sparse, norm="l2")
    sparse_expected = Normalizer(norm="l2").fit(X_sparse).transform(X_sparse)
    assert np.allclose(sparse_result.toarray(), sparse_expected.toarray())


def test_scale_matches_sklearn_dense_axes_and_options() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import scale

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [np.nan, 2.0, 1.0]], dtype=np.float64)
    assert np.allclose(scale(X, axis=0), sklearn_scale(X, axis=0), equal_nan=True)
    assert np.allclose(scale(X, axis=1), sklearn_scale(X, axis=1), equal_nan=True)
    assert np.allclose(
        scale(X, axis=0, with_mean=False, with_std=True),
        sklearn_scale(X, axis=0, with_mean=False, with_std=True),
        equal_nan=True,
    )
    assert np.allclose(
        scale(X, axis=1, with_mean=True, with_std=False),
        sklearn_scale(X, axis=1, with_mean=True, with_std=False),
        equal_nan=True,
    )


def test_scale_matches_sklearn_sparse_axis0_without_centering() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import scale

    X = sp.csc_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    result = scale(X, axis=0, with_mean=False, with_std=True)
    expected = sklearn_scale(X, axis=0, with_mean=False, with_std=True)
    assert np.allclose(result.toarray(), expected.toarray())


def test_scale_sparse_centering_and_axis_errors_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import scale

    X = sp.csc_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="Cannot center sparse matrices"):
        scale(X, with_mean=True)
    with pytest.raises(ValueError, match="Cannot center sparse matrices"):
        sklearn_scale(X, with_mean=True)
    with pytest.raises(ValueError, match="axis=0"):
        scale(X, axis=1, with_mean=False)
    with pytest.raises(ValueError, match="axis=0"):
        sklearn_scale(X, axis=1, with_mean=False)
