from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.preprocessing import Binarizer, KernelCenterer, MaxAbsScaler, Normalizer
from sklearn.preprocessing import add_dummy_feature as sklearn_add_dummy_feature
from sklearn.preprocessing import binarize as sklearn_binarize
from sklearn.preprocessing import maxabs_scale as sklearn_maxabs_scale
from sklearn.preprocessing import minmax_scale as sklearn_minmax_scale
from sklearn.preprocessing import normalize as sklearn_normalize
from sklearn.preprocessing import robust_scale as sklearn_robust_scale
from sklearn.preprocessing import scale as sklearn_scale


def test_preprocessing_basic_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        add_dummy_feature,
        binarize,
        binarizer_transform,
        kernel_centerer_fit,
        kernel_centerer_transform,
        maxabs_scale,
        maxabs_scaler_fit,
        maxabs_scaler_inverse_transform,
        maxabs_scaler_partial_fit,
        maxabs_scaler_transform,
        minmax_scale,
        normalize,
        normalizer_transform,
        robust_scale,
        scale,
    )

    assert callable(add_dummy_feature)
    assert callable(binarize)
    assert callable(binarizer_transform)
    assert callable(kernel_centerer_fit)
    assert callable(kernel_centerer_transform)
    assert callable(maxabs_scale)
    assert callable(maxabs_scaler_fit)
    assert callable(maxabs_scaler_inverse_transform)
    assert callable(maxabs_scaler_partial_fit)
    assert callable(maxabs_scaler_transform)
    assert callable(minmax_scale)
    assert callable(normalize)
    assert callable(normalizer_transform)
    assert callable(robust_scale)
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


def test_kernel_centerer_fit_matches_sklearn_state() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import kernel_centerer_fit

    K = np.array([[9.0, 2.0, -2.0], [2.0, 14.0, -13.0], [-2.0, -13.0, 21.0]], dtype=np.float64)
    state = kernel_centerer_fit(K)
    expected = KernelCenterer().fit(K)

    assert np.allclose(state.k_fit_rows, expected.K_fit_rows_)
    assert np.isclose(state.k_fit_all, expected.K_fit_all_)
    assert state.n_features_in == expected.n_features_in_


def test_kernel_centerer_transform_matches_sklearn_square_and_rectangular() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import kernel_centerer_fit, kernel_centerer_transform

    K_train = np.array([[9.0, 2.0, -2.0], [2.0, 14.0, -13.0], [-2.0, -13.0, 21.0]], dtype=np.float64)
    state = kernel_centerer_fit(K_train)
    transformer = KernelCenterer().fit(K_train)
    assert np.allclose(kernel_centerer_transform(K_train, state), transformer.transform(K_train))

    K_test = np.array([[1.0, 4.0, 5.0], [2.0, -1.0, 3.0]], dtype=np.float64)
    assert np.allclose(kernel_centerer_transform(K_test, state), transformer.transform(K_test))


def test_kernel_centerer_errors_match_sklearn_shape_requirements() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import kernel_centerer_fit, kernel_centerer_transform

    K_rectangular = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    with pytest.raises(Exception, match="square"):
        kernel_centerer_fit(K_rectangular)
    with pytest.raises(ValueError, match="square"):
        KernelCenterer().fit(K_rectangular)

    state = kernel_centerer_fit(np.eye(3, dtype=np.float64))
    with pytest.raises(Exception, match="columns"):
        kernel_centerer_transform(np.ones((2, 2), dtype=np.float64), state)
    with pytest.raises(ValueError):
        KernelCenterer().fit(np.eye(3, dtype=np.float64)).transform(np.ones((2, 2), dtype=np.float64))


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


def test_maxabs_scale_matches_sklearn_dense_axes_and_1d() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import maxabs_scale

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [np.nan, 2.0, 0.0]], dtype=np.float64)
    assert np.allclose(maxabs_scale(X, axis=0), sklearn_maxabs_scale(X, axis=0), equal_nan=True)
    assert np.allclose(maxabs_scale(X, axis=1), sklearn_maxabs_scale(X, axis=1), equal_nan=True)

    x = np.array([-2.0, 0.0, 4.0], dtype=np.float64)
    assert np.allclose(maxabs_scale(x), sklearn_maxabs_scale(x))


def test_maxabs_scale_matches_sklearn_sparse_axes() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import maxabs_scale

    X = sp.csr_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    for axis in (0, 1):
        result = maxabs_scale(X, axis=axis)
        expected = sklearn_maxabs_scale(X, axis=axis)
        assert np.allclose(result.toarray(), expected.toarray())


def test_maxabs_scaler_fit_matches_sklearn_state_dense_and_sparse() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import maxabs_scaler_fit

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [np.nan, 2.0, 0.0]], dtype=np.float64)
    state = maxabs_scaler_fit(X)
    expected = MaxAbsScaler().fit(X)
    assert np.allclose(state.scale, expected.scale_, equal_nan=True)
    assert np.allclose(state.max_abs, expected.max_abs_, equal_nan=True)
    assert state.n_features_in == expected.n_features_in_
    assert state.n_samples_seen == expected.n_samples_seen_

    X_sparse = sp.csr_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    sparse_state = maxabs_scaler_fit(X_sparse)
    sparse_expected = MaxAbsScaler().fit(X_sparse)
    assert np.allclose(sparse_state.scale, sparse_expected.scale_)
    assert np.allclose(sparse_state.max_abs, sparse_expected.max_abs_)


def test_maxabs_scaler_partial_fit_accumulates_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import maxabs_scaler_partial_fit

    X1 = np.array([[1.0, -2.0, 0.0], [0.0, 1.0, 3.0]], dtype=np.float64)
    X2 = np.array([[4.0, -1.0, -5.0]], dtype=np.float64)
    state = maxabs_scaler_partial_fit(X1)
    state = maxabs_scaler_partial_fit(X2, state)

    expected = MaxAbsScaler().partial_fit(X1).partial_fit(X2)
    assert np.allclose(state.scale, expected.scale_)
    assert np.allclose(state.max_abs, expected.max_abs_)
    assert state.n_samples_seen == expected.n_samples_seen_


def test_maxabs_scaler_transform_inverse_and_clip_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        maxabs_scaler_fit,
        maxabs_scaler_inverse_transform,
        maxabs_scaler_transform,
    )

    X = np.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]], dtype=np.float64)
    X_test = np.array([[4.0, -2.0, 4.0], [1.0, 0.5, -3.0]], dtype=np.float64)
    state = maxabs_scaler_fit(X)
    scaler = MaxAbsScaler().fit(X)

    transformed = maxabs_scaler_transform(X_test, state)
    expected = scaler.transform(X_test)
    assert np.allclose(transformed, expected)
    assert np.allclose(maxabs_scaler_inverse_transform(transformed, state), scaler.inverse_transform(expected))

    clipped = maxabs_scaler_transform(X_test, state, clip=True)
    expected_clipped = MaxAbsScaler(clip=True).fit(X).transform(X_test)
    assert np.allclose(clipped, expected_clipped)


def test_maxabs_scaler_sparse_transform_inverse_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import (
        maxabs_scaler_fit,
        maxabs_scaler_inverse_transform,
        maxabs_scaler_transform,
    )

    X = sp.csr_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    state = maxabs_scaler_fit(X)
    scaler = MaxAbsScaler().fit(X)
    result = maxabs_scaler_transform(X, state)
    expected = scaler.transform(X)
    assert np.allclose(result.toarray(), expected.toarray())
    assert np.allclose(maxabs_scaler_inverse_transform(result, state).toarray(), scaler.inverse_transform(expected).toarray())


def test_maxabs_scaler_feature_mismatch_rejected_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import maxabs_scaler_fit, maxabs_scaler_transform

    state = maxabs_scaler_fit(np.eye(3, dtype=np.float64))
    with pytest.raises(Exception, match="feature count"):
        maxabs_scaler_transform(np.ones((2, 2), dtype=np.float64), state)
    with pytest.raises(ValueError):
        MaxAbsScaler().fit(np.eye(3, dtype=np.float64)).transform(np.ones((2, 2), dtype=np.float64))


def test_minmax_scale_matches_sklearn_dense_axes_and_range() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import minmax_scale

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [np.nan, 2.0, 0.0]], dtype=np.float64)
    assert np.allclose(minmax_scale(X, axis=0), sklearn_minmax_scale(X, axis=0), equal_nan=True)
    assert np.allclose(
        minmax_scale(X, feature_range=(-1.0, 2.0), axis=1),
        sklearn_minmax_scale(X, feature_range=(-1.0, 2.0), axis=1),
        equal_nan=True,
    )

    x = np.array([-2.0, 0.0, 4.0], dtype=np.float64)
    assert np.allclose(minmax_scale(x, feature_range=(-2.0, 3.0)), sklearn_minmax_scale(x, feature_range=(-2.0, 3.0)))


def test_minmax_scale_rejects_invalid_range_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import minmax_scale

    X = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    with pytest.raises(Exception):
        minmax_scale(X, feature_range=(1.0, 1.0))
    with pytest.raises(ValueError):
        sklearn_minmax_scale(X, feature_range=(1.0, 1.0))


def test_robust_scale_matches_sklearn_dense_axes_options_and_1d() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import robust_scale

    X = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [10.0, 2.0, np.nan]], dtype=np.float64)
    assert np.allclose(robust_scale(X, axis=0), sklearn_robust_scale(X, axis=0), equal_nan=True)
    assert np.allclose(
        robust_scale(X, axis=1, quantile_range=(10.0, 90.0), unit_variance=True),
        sklearn_robust_scale(X, axis=1, quantile_range=(10.0, 90.0), unit_variance=True),
        equal_nan=True,
    )
    assert np.allclose(
        robust_scale(X, axis=0, with_centering=False, with_scaling=True),
        sklearn_robust_scale(X, axis=0, with_centering=False, with_scaling=True),
        equal_nan=True,
    )

    x = np.array([-2.0, 0.0, 4.0], dtype=np.float64)
    assert np.allclose(robust_scale(x), sklearn_robust_scale(x))


def test_robust_scale_matches_sklearn_sparse_without_centering() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import robust_scale

    X = sp.csr_matrix([[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    for axis in (0, 1):
        result = robust_scale(X, axis=axis, with_centering=False)
        expected = sklearn_robust_scale(X, axis=axis, with_centering=False)
        assert np.allclose(result.toarray(), expected.toarray())


def test_robust_scale_sparse_centering_error_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.preprocessing import robust_scale

    X = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="Cannot center sparse matrices"):
        robust_scale(X, with_centering=True)
    with pytest.raises(ValueError, match="Cannot center sparse matrices"):
        sklearn_robust_scale(X, with_centering=True)
