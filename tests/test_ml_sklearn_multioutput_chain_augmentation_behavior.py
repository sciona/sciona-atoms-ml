from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from sciona.atoms.ml.sklearn.multioutput.chain_augmentation import (
    chain_cv_feature_column,
    chain_dense_cv_feature_buffer,
    chain_sparse_cv_feature_buffer,
    chain_sparse_step_features,
    chain_sparse_training_features,
)


def test_dense_cv_feature_buffer_appends_zero_columns() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    observed = chain_dense_cv_feature_buffer(X, 3)

    expected = np.array(
        [[1.0, 2.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    assert np.array_equal(observed, expected)


def test_sparse_cv_feature_buffer_appends_sparse_zero_columns() -> None:
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))

    observed = chain_sparse_cv_feature_buffer(X, 2)

    assert sp.issparse(observed)
    assert observed.shape == (2, 4)
    assert np.array_equal(
        observed.toarray(),
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]], dtype=np.float64),
    )


def test_sparse_cv_feature_buffer_converts_dok_array_before_hstack() -> None:
    if not hasattr(sp, "dok_array"):
        return

    X = sp.dok_array(np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64))

    observed = chain_sparse_cv_feature_buffer(X, 1)

    assert sp.issparse(observed)
    assert observed.shape == (2, 3)
    assert np.array_equal(
        observed.toarray(),
        np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
    )


def test_sparse_training_features_appends_ordered_targets_and_returns_csr() -> None:
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
    Y_ordered = np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float64)

    observed = chain_sparse_training_features(X, Y_ordered)

    assert sp.issparse(observed)
    assert observed.shape == (2, 4)
    assert np.array_equal(
        observed.toarray(),
        np.array([[1.0, 0.0, 9.0, 8.0], [0.0, 2.0, 7.0, 6.0]], dtype=np.float64),
    )


def test_sparse_step_features_appends_previous_predictions() -> None:
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
    previous_predictions = np.array([[0.25], [0.75]], dtype=np.float64)

    observed = chain_sparse_step_features(X, previous_predictions)

    assert sp.issparse(observed)
    assert observed.shape == (2, 3)
    assert np.allclose(
        observed.toarray(),
        np.array([[1.0, 0.0, 0.25], [0.0, 2.0, 0.75]], dtype=np.float64),
    )


def test_chain_cv_feature_column_keeps_1d_outputs() -> None:
    cv_result = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    observed = chain_cv_feature_column(cv_result)

    assert np.array_equal(observed, cv_result)


def test_chain_cv_feature_column_selects_positive_class_probability() -> None:
    cv_result = np.array(
        [[0.8, 0.2], [0.1, 0.9], [0.4, 0.6]],
        dtype=np.float64,
    )

    observed = chain_cv_feature_column(cv_result)

    assert np.allclose(observed, np.array([0.2, 0.9, 0.6], dtype=np.float64))
