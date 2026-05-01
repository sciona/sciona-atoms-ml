from __future__ import annotations

import numpy as np
import pytest


def test_dictionary_learning_wrapper_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_wrapper_bookkeeping import (
        dict_learning_online_return_values,
        dict_learning_return_values,
        dictionary_learning_lasso_method,
        dictionary_learning_resolved_n_components,
    )

    assert callable(dictionary_learning_lasso_method)
    assert callable(dictionary_learning_resolved_n_components)
    assert callable(dict_learning_return_values)
    assert callable(dict_learning_online_return_values)


def test_dictionary_learning_wrapper_bookkeeping_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_wrapper_bookkeeping import (
        dict_learning_online_return_values,
        dict_learning_return_values,
        dictionary_learning_lasso_method,
        dictionary_learning_resolved_n_components,
    )

    assert dictionary_learning_lasso_method("lars") == "lasso_lars"
    assert dictionary_learning_lasso_method("cd") == "lasso_cd"

    X = np.ones((4, 7), dtype=np.float64)
    assert dictionary_learning_resolved_n_components(X, None) == 7
    assert dictionary_learning_resolved_n_components(X, 3) == 3

    code = np.arange(12.0, dtype=np.float64).reshape(4, 3)
    components = np.arange(21.0, dtype=np.float64).reshape(3, 7)
    errors = np.array([3.0, 2.0, 1.0], dtype=np.float64)

    assert dict_learning_return_values(code, components, errors, 5, False) == (
        code,
        components,
        errors,
    )
    assert dict_learning_return_values(code, components, errors, 5, True) == (
        code,
        components,
        errors,
        5,
    )

    assert np.array_equal(
        dict_learning_online_return_values(components, False),
        components,
    )
    returned_code, returned_components = dict_learning_online_return_values(
        components,
        True,
        code=code,
    )
    assert np.array_equal(returned_code, code)
    assert np.array_equal(returned_components, components)


def test_dictionary_learning_wrapper_bookkeeping_contracts() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_wrapper_bookkeeping import (
        dict_learning_online_return_values,
        dict_learning_return_values,
        dictionary_learning_lasso_method,
        dictionary_learning_resolved_n_components,
    )

    with pytest.raises(Exception):
        dictionary_learning_lasso_method("omp")

    with pytest.raises(Exception):
        dictionary_learning_resolved_n_components(np.ones((2, 3)), 0)

    code = np.ones((4, 3), dtype=np.float64)
    components = np.ones((2, 7), dtype=np.float64)
    errors = np.ones(3, dtype=np.float64)

    with pytest.raises(Exception):
        dict_learning_return_values(code, components, errors, 2, False)

    with pytest.raises(Exception):
        dict_learning_online_return_values(components, True, code=code)
