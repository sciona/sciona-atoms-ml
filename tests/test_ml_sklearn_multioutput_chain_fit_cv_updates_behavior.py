from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.multioutput.chain_fit_cv_updates import (
    chain_fit_cv_update_required,
    chain_fit_dense_cv_feature_update,
    chain_fit_feature_column_index,
    chain_fit_sparse_cv_feature_update,
)


def test_chain_fit_cv_update_required_matches_nonterminal_step_rule() -> None:
    assert chain_fit_cv_update_required(0, 3) is True
    assert chain_fit_cv_update_required(1, 3) is True
    assert chain_fit_cv_update_required(2, 3) is False


def test_chain_fit_feature_column_index_matches_sklearn_formula() -> None:
    assert chain_fit_feature_column_index(4, 0) == 4
    assert chain_fit_feature_column_index(4, 2) == 6


def test_chain_fit_dense_cv_feature_update_matches_dense_assignment() -> None:
    X_aug = np.array(
        [[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0], [5.0, 6.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    cv_column = np.array([0.2, 0.7, 0.5], dtype=np.float64)
    col_idx = 2

    expected = X_aug.copy()
    expected[:, col_idx] = cv_column

    observed = chain_fit_dense_cv_feature_update(X_aug, cv_column, col_idx)

    assert np.allclose(observed, expected)
    assert np.allclose(X_aug[:, :2], observed[:, :2])
    assert np.allclose(X_aug[:, 3], observed[:, 3])


def test_chain_fit_sparse_cv_feature_update_matches_sparse_assignment() -> None:
    X_aug = sp.lil_matrix(
        np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    )
    cv_column = np.array([0.1, 0.8, 0.6], dtype=np.float64)
    col_idx = 3

    expected = X_aug.copy()
    expected[:, col_idx] = np.expand_dims(cv_column, 1)

    observed = chain_fit_sparse_cv_feature_update(X_aug, cv_column, col_idx)

    assert sp.issparse(observed)
    assert np.allclose(observed.toarray(), expected.toarray())


def test_chain_fit_sparse_cv_feature_update_supports_sparse_array() -> None:
    if not hasattr(sp, "lil_array"):
        return

    X_aug = sp.lil_array(
        np.array(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float64,
        )
    )
    cv_column = np.array([0.3, 0.9], dtype=np.float64)

    observed = chain_fit_sparse_cv_feature_update(X_aug, cv_column, 2)

    assert sp.issparse(observed)
    assert np.allclose(
        observed.toarray(),
        np.array([[1.0, 0.0, 0.3], [0.0, 2.0, 0.9]], dtype=np.float64),
    )


def test_chain_fit_cv_update_atoms_reject_invalid_inputs() -> None:
    with pytest.raises((ValueError, ViolationError)):
        chain_fit_cv_update_required(2, 2)

    with pytest.raises((ValueError, ViolationError)):
        chain_fit_feature_column_index(0, 1)

    with pytest.raises((ValueError, ViolationError)):
        chain_fit_dense_cv_feature_update(
            np.array([[1.0, 0.0]], dtype=np.float64),
            np.array([0.2, 0.3], dtype=np.float64),
            1,
        )
