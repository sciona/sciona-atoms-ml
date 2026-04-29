from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.multioutput.chain_fit_step_data import (
    chain_fit_dense_step_features,
    chain_fit_sparse_step_features,
    chain_fit_step_feature_limit,
    chain_fit_target_column,
)


def test_chain_fit_step_feature_limit_matches_sklearn_formula() -> None:
    assert chain_fit_step_feature_limit(4, 0) == 4
    assert chain_fit_step_feature_limit(4, 2) == 6


def test_chain_fit_target_column_matches_ordered_target_selection() -> None:
    Y = np.array(
        [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0]],
        dtype=np.float64,
    )
    order = np.array([2, 0, 1], dtype=np.int64)

    observed = chain_fit_target_column(Y, order, 1)

    assert np.array_equal(observed, Y[:, order[1]])


def test_chain_fit_dense_step_features_match_prefix_slice() -> None:
    X_aug = np.array(
        [[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 11.0, 21.0], [5.0, 6.0, 12.0, 22.0]],
        dtype=np.float64,
    )

    observed = chain_fit_dense_step_features(X_aug, 3)

    assert np.array_equal(observed, X_aug[:, :3])


def test_chain_fit_sparse_step_features_match_prefix_slice() -> None:
    X_aug = sp.lil_matrix(
        np.array(
            [[1.0, 0.0, 10.0, 20.0], [0.0, 2.0, 11.0, 21.0], [3.0, 4.0, 12.0, 22.0]],
            dtype=np.float64,
        )
    )

    observed = chain_fit_sparse_step_features(X_aug, 2)

    assert sp.issparse(observed)
    assert np.allclose(observed.toarray(), X_aug[:, :2].toarray())


def test_chain_fit_sparse_step_features_support_sparse_array() -> None:
    if not hasattr(sp, "csr_array"):
        return
    X_aug = sp.csr_array(
        np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 11.0]], dtype=np.float64)
    )

    observed = chain_fit_sparse_step_features(X_aug, 1)

    assert sp.issparse(observed)
    assert np.allclose(observed.toarray(), np.array([[1.0], [3.0]], dtype=np.float64))


def test_chain_fit_step_data_atoms_reject_invalid_inputs() -> None:
    with pytest.raises((ValueError, ViolationError)):
        chain_fit_step_feature_limit(0, 1)

    with pytest.raises((ValueError, ViolationError)):
        chain_fit_target_column(
            np.array([[1.0, 2.0]], dtype=np.float64),
            np.array([0, 0], dtype=np.int64),
            0,
        )

    with pytest.raises((ValueError, ViolationError)):
        chain_fit_dense_step_features(np.array([[1.0, 2.0]], dtype=np.float64), 3)
