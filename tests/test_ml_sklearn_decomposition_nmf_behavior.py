from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from icontract import ViolationError
from sklearn.decomposition._nmf import (
    _beta_divergence,
    _beta_loss_to_float,
    _check_init,
    _initialize_nmf,
    trace_dot,
)
from sklearn.utils.extmath import randomized_svd


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


def _init_matrix() -> np.ndarray:
    return np.array([[0.4, 0.2], [0.1, 0.9], [0.8, 0.3]], dtype=np.float64)


def _nndsvd_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    X = np.array(
        [
            [1.0, 0.0, 2.0, 0.5],
            [0.5, 1.5, 0.3, 0.2],
            [2.1, 0.4, 1.0, 1.2],
        ],
        dtype=np.float64,
    )
    U, S, V = randomized_svd(X, 2, random_state=11)
    return X, U, S, V


def test_nmf_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf import (
        nmf_beta_divergence,
        nmf_beta_loss_to_float,
        nmf_check_init_matrix,
        nmf_nndsvd_from_svd,
        nmf_random_initialize,
        nmf_trace_dot,
    )

    assert callable(nmf_beta_divergence)
    assert callable(nmf_beta_loss_to_float)
    assert callable(nmf_check_init_matrix)
    assert callable(nmf_nndsvd_from_svd)
    assert callable(nmf_random_initialize)
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


def test_nmf_random_initialize_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf import nmf_random_initialize

    X = np.array([[1.0, 0.5, 2.0], [0.3, 1.2, 0.7], [2.0, 0.1, 1.5]], dtype=np.float64)
    expected_w, expected_h = _initialize_nmf(X, 2, init="random", random_state=17)
    actual_w, actual_h = nmf_random_initialize(X, 2, random_state=17)

    np.testing.assert_allclose(actual_w, expected_w)
    np.testing.assert_allclose(actual_h, expected_h)


@pytest.mark.parametrize("init", ["nndsvd", "nndsvda", "nndsvdar"])
def test_nmf_nndsvd_from_svd_matches_sklearn(init: str) -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf import nmf_nndsvd_from_svd

    X, U, S, V = _nndsvd_fixture()
    expected_w, expected_h = _initialize_nmf(X, 2, init=init, random_state=11)
    actual_w, actual_h = nmf_nndsvd_from_svd(U, S, V, init, float(X.mean()), random_state=11)

    np.testing.assert_allclose(actual_w, expected_w)
    np.testing.assert_allclose(actual_h, expected_h)


def test_nmf_check_init_matrix_matches_sklearn_validation_contract() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf import nmf_check_init_matrix

    A = _init_matrix()
    _check_init(A, ("auto", 2), "W")
    validated = nmf_check_init_matrix(A, ("auto", 2), "W")

    np.testing.assert_allclose(validated, A)
    assert validated.dtype == np.float64


def test_contracts_reject_invalid_nmf_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf import (
        nmf_beta_divergence,
        nmf_beta_loss_to_float,
        nmf_check_init_matrix,
        nmf_nndsvd_from_svd,
        nmf_random_initialize,
        nmf_trace_dot,
    )

    X, W, H = _factors()
    U, S, V = randomized_svd(X, 2, random_state=0)

    with pytest.raises(ViolationError):
        nmf_beta_loss_to_float("bad-loss")

    with pytest.raises(ViolationError):
        nmf_trace_dot(X, X[:, :2])

    with pytest.raises(ViolationError):
        nmf_beta_divergence(X, W[:, :1], H, "frobenius")

    with pytest.raises(ViolationError):
        nmf_beta_divergence(X - 2.0, W, H, "frobenius")

    with pytest.raises(ViolationError):
        nmf_random_initialize(X - 2.0, 2, random_state=0)

    with pytest.raises(ViolationError):
        nmf_nndsvd_from_svd(U, S, V, "bad-init", float(X.mean()))

    with pytest.raises(ViolationError):
        nmf_check_init_matrix(X, (0, 3), "W")

    X_with_zero = X.copy()
    X_with_zero[0, 0] = 0.0
    with pytest.raises(ViolationError):
        nmf_beta_divergence(X_with_zero, W, H, "itakura-saito")


def test_nmf_check_init_matrix_rejects_shape_mismatch_and_zero_matrix() -> None:
    from sciona.atoms.ml.sklearn.decomposition.nmf import nmf_check_init_matrix

    with pytest.raises(ValueError, match="wrong first dimension"):
        nmf_check_init_matrix(_init_matrix(), (2, 2), "W")

    with pytest.raises(ValueError, match="full of zeros"):
        nmf_check_init_matrix(np.zeros((3, 2), dtype=np.float64), ("auto", 2), "W")
