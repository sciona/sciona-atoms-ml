from __future__ import annotations

import numpy as np
import pytest


def test_dictionary_learning_postfit_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_postfit_shell import (
        dictionary_learning_fit_components,
        dictionary_learning_fit_errors,
        dictionary_learning_fit_n_iter,
        dictionary_learning_fit_transform_output,
    )

    assert callable(dictionary_learning_fit_components)
    assert callable(dictionary_learning_fit_errors)
    assert callable(dictionary_learning_fit_n_iter)
    assert callable(dictionary_learning_fit_transform_output)


def test_dictionary_learning_postfit_shell_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_postfit_shell import (
        dictionary_learning_fit_components,
        dictionary_learning_fit_errors,
        dictionary_learning_fit_n_iter,
        dictionary_learning_fit_transform_output,
    )

    code = np.arange(12.0, dtype=np.float64).reshape(4, 3)
    dictionary = np.arange(15.0, dtype=np.float64).reshape(3, 5)
    errors = np.array([3.0, 2.0, 1.0], dtype=np.float64)

    assert np.array_equal(dictionary_learning_fit_transform_output(code), code)
    assert np.array_equal(dictionary_learning_fit_components(dictionary), dictionary)
    assert np.array_equal(dictionary_learning_fit_errors(errors), errors)
    assert dictionary_learning_fit_n_iter(7) == 7


def test_dictionary_learning_postfit_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_postfit_shell import (
        dictionary_learning_fit_components,
        dictionary_learning_fit_errors,
        dictionary_learning_fit_n_iter,
        dictionary_learning_fit_transform_output,
    )

    with pytest.raises(Exception):
        dictionary_learning_fit_transform_output(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(Exception):
        dictionary_learning_fit_components(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises(Exception):
        dictionary_learning_fit_errors(np.array([[1.0]], dtype=np.float64))

    with pytest.raises(Exception):
        dictionary_learning_fit_n_iter(0)
