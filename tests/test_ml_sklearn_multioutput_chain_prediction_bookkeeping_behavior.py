from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from sciona.atoms.ml.sklearn.multioutput.chain_prediction_bookkeeping import (
    chain_prediction_feature_buffer,
    chain_prediction_method_name,
    chain_prediction_previous_predictions,
    chain_prediction_output_buffer,
    chain_sparse_hstack_base,
)


def test_chain_prediction_method_name_defaults_to_predict() -> None:
    assert chain_prediction_method_name() == "predict"


def test_chain_prediction_method_name_keeps_configured_name() -> None:
    assert chain_prediction_method_name("predict_proba") == "predict_proba"


def test_chain_prediction_output_buffer_is_zero_initialized() -> None:
    observed = chain_prediction_output_buffer(3, 2)

    assert observed.shape == (3, 2)
    assert observed.dtype == np.float64
    assert np.array_equal(observed, np.zeros((3, 2), dtype=np.float64))


def test_chain_prediction_feature_buffer_is_zero_initialized() -> None:
    observed = chain_prediction_feature_buffer(2, 3)

    assert observed.shape == (2, 3)
    assert observed.dtype == np.float64
    assert np.array_equal(observed, np.zeros((2, 3), dtype=np.float64))


def test_chain_prediction_previous_predictions_returns_empty_prefix_for_first_step() -> None:
    feature_chain = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)

    observed = chain_prediction_previous_predictions(feature_chain, 0)

    assert observed.shape == (2, 0)
    assert observed.dtype == np.float64


def test_chain_prediction_previous_predictions_returns_filled_prefix() -> None:
    feature_chain = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)

    observed = chain_prediction_previous_predictions(feature_chain, 2)

    assert np.array_equal(
        observed,
        np.array([[0.1, 0.2], [0.4, 0.5]], dtype=np.float64),
    )


def test_chain_sparse_hstack_base_converts_dok_array_to_coo_array() -> None:
    if not hasattr(sp, "dok_array"):
        return

    X = sp.dok_array(np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64))

    observed = chain_sparse_hstack_base(X)

    assert sp.issparse(observed)
    assert not sp.isspmatrix(observed)
    assert getattr(observed, "format", None) == "coo"
    assert np.array_equal(observed.toarray(), X.toarray())


def test_chain_sparse_hstack_base_leaves_sparse_matrix_unchanged() -> None:
    X = sp.dok_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))

    observed = chain_sparse_hstack_base(X)

    assert observed is X


def test_chain_sparse_hstack_base_leaves_csr_array_unchanged() -> None:
    if not hasattr(sp, "csr_array"):
        return

    X = sp.csr_array(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))

    observed = chain_sparse_hstack_base(X)

    assert observed is X
