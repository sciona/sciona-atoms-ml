from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.decomposition._dict_learning import _update_dict


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Y = np.array(
        [
            [1.0, 0.5, 2.0],
            [0.0, 1.5, 0.5],
            [2.0, 1.0, 1.0],
            [1.5, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    code = np.array(
        [
            [1.0, 0.2],
            [0.5, 1.0],
            [1.2, 0.4],
            [0.3, 1.3],
        ],
        dtype=np.float64,
    )
    dictionary = np.array(
        [
            [0.8, 0.2, 0.3],
            [0.1, 0.7, 0.4],
        ],
        dtype=np.float64,
    )
    return Y, code, dictionary


def test_dictionary_update_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_update import (
        dictionary_learning_active_update,
        dictionary_learning_sufficient_statistics,
    )

    assert callable(dictionary_learning_sufficient_statistics)
    assert callable(dictionary_learning_active_update)


def test_sufficient_statistics_match_sklearn_setup() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_update import dictionary_learning_sufficient_statistics

    Y, code, _ = _data()
    A, B = dictionary_learning_sufficient_statistics(Y, code)

    assert np.allclose(A, code.T @ code)
    assert np.allclose(B, Y.T @ code)
    assert np.allclose(A, A.T)


def test_active_update_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_update import (
        dictionary_learning_active_update,
        dictionary_learning_sufficient_statistics,
    )

    Y, code, dictionary = _data()
    A, B = dictionary_learning_sufficient_statistics(Y, code)
    result = dictionary_learning_active_update(dictionary, A, B)

    expected_dictionary = dictionary.copy()
    expected_code = code.copy()
    _update_dict(expected_dictionary, Y, expected_code, A=A.copy(), B=B.copy(), random_state=0)

    assert np.allclose(result, expected_dictionary)
    assert np.all(np.linalg.norm(result, axis=1) <= 1.0 + 1e-12)
    assert np.array_equal(expected_code, code)


def test_active_update_positive_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_update import (
        dictionary_learning_active_update,
        dictionary_learning_sufficient_statistics,
    )

    Y, code, dictionary = _data()
    dictionary = dictionary - 0.6
    A, B = dictionary_learning_sufficient_statistics(Y, code)
    result = dictionary_learning_active_update(dictionary, A, B, positive=True)

    expected_dictionary = dictionary.copy()
    _update_dict(expected_dictionary, Y, code.copy(), A=A.copy(), B=B.copy(), random_state=0, positive=True)

    assert np.allclose(result, expected_dictionary)
    assert np.all(result >= 0.0)


def test_active_update_accepts_explicit_statistics() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_update import dictionary_learning_active_update

    dictionary = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    A = np.array([[2.0, 0.1], [0.1, 1.5]], dtype=np.float64)
    B = np.array([[1.0, 0.5], [0.4, 1.3]], dtype=np.float64)

    result = dictionary_learning_active_update(dictionary, A, B)

    assert result.shape == dictionary.shape
    assert np.all(np.isfinite(result))
    assert np.all(np.linalg.norm(result, axis=1) <= 1.0 + 1e-12)


def test_contracts_reject_invalid_dictionary_update_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_update import (
        dictionary_learning_active_update,
        dictionary_learning_sufficient_statistics,
    )

    Y, code, dictionary = _data()
    A, B = dictionary_learning_sufficient_statistics(Y, code)

    with pytest.raises(ViolationError):
        dictionary_learning_sufficient_statistics(Y, code[:2])

    with pytest.raises(ViolationError):
        dictionary_learning_active_update(dictionary, A[:1], B)

    with pytest.raises(ViolationError):
        dictionary_learning_active_update(dictionary, np.diag([1e-8, 1.0]), B)

    with pytest.raises(ViolationError):
        dictionary_learning_active_update(dictionary, A, B[:2])
