from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg
from sklearn.utils.extmath import svd_flip


def test_dictionary_learning_loop_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_loop import (
        dictionary_learning_callback_due,
        dictionary_learning_converged,
        dictionary_learning_cost,
        dictionary_learning_resize_factors,
        dictionary_learning_svd_initialize,
    )

    assert callable(dictionary_learning_svd_initialize)
    assert callable(dictionary_learning_resize_factors)
    assert callable(dictionary_learning_cost)
    assert callable(dictionary_learning_converged)
    assert callable(dictionary_learning_callback_due)


def test_dictionary_learning_svd_initialize_matches_source_formula() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_loop import dictionary_learning_svd_initialize

    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, 1.0, 1.5],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    code, singular_values, dictionary = linalg.svd(X, full_matrices=False)
    code, dictionary = svd_flip(code, dictionary)
    expected_code = code
    expected_dictionary = singular_values[:, np.newaxis] * dictionary

    result_code, result_dictionary = dictionary_learning_svd_initialize(X)

    assert np.allclose(result_code, expected_code)
    assert np.allclose(result_dictionary, expected_dictionary)


@pytest.mark.parametrize("n_components", [2, 4])
def test_dictionary_learning_resize_factors_matches_source_branches(n_components: int) -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_loop import dictionary_learning_resize_factors

    code = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    dictionary = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64)
    rank = dictionary.shape[0]

    if n_components <= rank:
        expected_code = code[:, :n_components]
        expected_dictionary = dictionary[:n_components, :]
    else:
        expected_code = np.c_[code, np.zeros((len(code), n_components - rank))]
        expected_dictionary = np.r_[dictionary, np.zeros((n_components - rank, dictionary.shape[1]))]

    result_code, result_dictionary = dictionary_learning_resize_factors(
        code,
        dictionary,
        n_components=n_components,
    )

    assert np.allclose(result_code, expected_code)
    assert np.allclose(result_dictionary, expected_dictionary)


def test_dictionary_learning_cost_matches_source_formula() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_loop import dictionary_learning_cost

    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    code = np.array([[0.5, 1.0], [1.5, 0.5]], dtype=np.float64)
    dictionary = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    alpha = 0.3

    expected = 0.5 * np.sum((X - code @ dictionary) ** 2) + alpha * np.sum(np.abs(code))
    assert dictionary_learning_cost(X, code, dictionary, alpha=alpha) == pytest.approx(expected)


def test_dictionary_learning_converged_matches_source_formula() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_loop import dictionary_learning_converged

    previous_cost = 10.0
    current_cost = 9.8
    assert dictionary_learning_converged(previous_cost, current_cost, tol=0.05) is True
    assert dictionary_learning_converged(previous_cost, current_cost, tol=0.001) is False


def test_dictionary_learning_callback_due_matches_source_cadence() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_loop import dictionary_learning_callback_due

    assert dictionary_learning_callback_due(0) is True
    assert dictionary_learning_callback_due(1) is False
    assert dictionary_learning_callback_due(5) is True
    assert dictionary_learning_callback_due(9) is False


def test_dictionary_learning_loop_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_loop import (
        dictionary_learning_callback_due,
        dictionary_learning_converged,
        dictionary_learning_cost,
        dictionary_learning_resize_factors,
        dictionary_learning_svd_initialize,
    )

    with pytest.raises(Exception):
        dictionary_learning_svd_initialize(np.array([[1.0, np.nan]], dtype=np.float64))
    with pytest.raises(Exception):
        dictionary_learning_resize_factors(np.ones((2, 2)), np.ones((3, 2)), n_components=2)
    with pytest.raises(Exception):
        dictionary_learning_cost(np.ones((2, 2)), np.ones((3, 2)), np.ones((2, 2)), alpha=0.1)
    with pytest.raises(Exception):
        dictionary_learning_converged(1.0, -1.0, tol=0.1)
    with pytest.raises(Exception):
        dictionary_learning_callback_due(-1)
