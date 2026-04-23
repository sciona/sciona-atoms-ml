from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from icontract import ViolationError
from sklearn.decomposition._nmf import _beta_divergence, _beta_loss_to_float, trace_dot


def _factors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array(
        [
            [1.0, 0.5, 2.0],
            [0.3, 1.2, 0.7],
            [2.0, 0.1, 1.5],
        ],
        dtype=np.float64,
    )
    W = np.array(
        [
            [0.6, 0.2],
            [0.1, 0.8],
            [0.9, 0.3],
        ],
        dtype=np.float64,
    )
    H = np.array([[1.3, 0.4, 1.1], [0.2, 1.0, 0.5]], dtype=np.float64)
    return X, W, H


def test_nmf_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf import (
        nmf_beta_divergence,
        nmf_beta_loss_to_float,
        nmf_trace_dot,
    )

    assert callable(nmf_beta_divergence)
    assert callable(nmf_beta_loss_to_float)
    assert callable(nmf_trace_dot)


def test_nmf_beta_loss_to_float_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf import nmf_beta_loss_to_float

    for beta_loss in ["frobenius", "kullback-leibler", "itakura-saito", 0.5, 3.0]:
        assert nmf_beta_loss_to_float(beta_loss) == float(_beta_loss_to_float(beta_loss))


def test_nmf_trace_dot_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf import nmf_trace_dot

    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    Y = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)

    assert np.isclose(nmf_trace_dot(X, Y), trace_dot(X, Y))
    assert np.isclose(nmf_trace_dot(sp.csr_matrix(X), sp.csr_matrix(Y)), trace_dot(X, Y))


def test_nmf_beta_divergence_matches_sklearn_dense_cases() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf import nmf_beta_divergence

    X, W, H = _factors()
    for beta in ["frobenius", "kullback-leibler", "itakura-saito", 0.5, 1.7, 3.0]:
        assert np.isclose(nmf_beta_divergence(X, W, H, beta), _beta_divergence(X, W, H, beta))


def test_nmf_beta_divergence_square_root_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf import nmf_beta_divergence

    X, W, H = _factors()
    for beta in ["frobenius", "kullback-leibler", 1.5]:
        assert np.isclose(
            nmf_beta_divergence(X, W, H, beta, square_root=True),
            _beta_divergence(X, W, H, beta, square_root=True),
        )


def test_nmf_beta_divergence_matches_sklearn_sparse_cases() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf import nmf_beta_divergence

    X, W, H = _factors()
    X_sparse = sp.csr_matrix(X * np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]))
    for beta in ["frobenius", "kullback-leibler", 1.5]:
        assert np.isclose(nmf_beta_divergence(X_sparse, W, H, beta), _beta_divergence(X_sparse, W, H, beta))


def test_contracts_reject_invalid_nmf_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf import (
        nmf_beta_divergence,
        nmf_beta_loss_to_float,
        nmf_trace_dot,
    )

    X, W, H = _factors()

    with pytest.raises(ViolationError):
        nmf_beta_loss_to_float("bad-loss")

    with pytest.raises(ViolationError):
        nmf_trace_dot(X, X[:, :2])

    with pytest.raises(ViolationError):
        nmf_beta_divergence(X, W[:, :1], H, "frobenius")

    with pytest.raises(ViolationError):
        nmf_beta_divergence(X - 2.0, W, H, "frobenius")

    X_with_zero = X.copy()
    X_with_zero[0, 0] = 0.0
    with pytest.raises(ViolationError):
        nmf_beta_divergence(X_with_zero, W, H, "itakura-saito")
