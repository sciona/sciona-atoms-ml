from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import MiniBatchDictionaryLearning


def test_dictionary_learning_minibatch_partial_fit_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell import (
        dictionary_learning_minibatch_partial_fit_components,
        dictionary_learning_minibatch_partial_fit_existing_dictionary,
        dictionary_learning_minibatch_partial_fit_first_call,
        dictionary_learning_minibatch_partial_fit_initial_dictionary,
        dictionary_learning_minibatch_partial_fit_initial_inner_stats,
        dictionary_learning_minibatch_partial_fit_initial_n_steps,
        dictionary_learning_minibatch_partial_fit_reset_required,
        dictionary_learning_minibatch_partial_fit_updated_n_steps,
    )

    assert callable(dictionary_learning_minibatch_partial_fit_first_call)
    assert callable(dictionary_learning_minibatch_partial_fit_reset_required)
    assert callable(dictionary_learning_minibatch_partial_fit_initial_n_steps)
    assert callable(dictionary_learning_minibatch_partial_fit_initial_inner_stats)
    assert callable(dictionary_learning_minibatch_partial_fit_initial_dictionary)
    assert callable(dictionary_learning_minibatch_partial_fit_existing_dictionary)
    assert callable(dictionary_learning_minibatch_partial_fit_components)
    assert callable(dictionary_learning_minibatch_partial_fit_updated_n_steps)


def test_dictionary_learning_minibatch_partial_fit_shell_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell import (
        dictionary_learning_minibatch_partial_fit_components,
        dictionary_learning_minibatch_partial_fit_existing_dictionary,
        dictionary_learning_minibatch_partial_fit_first_call,
        dictionary_learning_minibatch_partial_fit_initial_dictionary,
        dictionary_learning_minibatch_partial_fit_initial_inner_stats,
        dictionary_learning_minibatch_partial_fit_initial_n_steps,
        dictionary_learning_minibatch_partial_fit_reset_required,
        dictionary_learning_minibatch_partial_fit_updated_n_steps,
    )

    X = np.array([[1.0, 0.3, 0.5], [0.2, 0.4, 0.9]], dtype=np.float64)
    model = MiniBatchDictionaryLearning(n_components=2, random_state=0, batch_size=3)
    has_components = hasattr(model, "components_")

    assert has_components is False
    assert dictionary_learning_minibatch_partial_fit_first_call(has_components) is True
    assert dictionary_learning_minibatch_partial_fit_reset_required(has_components) is True
    assert dictionary_learning_minibatch_partial_fit_initial_n_steps(True) == 0

    model._check_params(X)
    random_state = np.random.RandomState(0)
    dictionary = model._initialize_dict(X, random_state)
    assert np.array_equal(dictionary_learning_minibatch_partial_fit_initial_dictionary(dictionary), dictionary)

    A, B = dictionary_learning_minibatch_partial_fit_initial_inner_stats(
        model._n_components, X.shape[1], X.dtype.name
    )
    assert A.shape == (model._n_components, model._n_components)
    assert B.shape == (X.shape[1], model._n_components)
    assert A.dtype == X.dtype
    assert B.dtype == X.dtype
    assert np.count_nonzero(A) == 0
    assert np.count_nonzero(B) == 0

    fitted = MiniBatchDictionaryLearning(n_components=2, random_state=0, batch_size=3).partial_fit(X)
    assert dictionary_learning_minibatch_partial_fit_first_call(True) is False
    assert dictionary_learning_minibatch_partial_fit_reset_required(True) is False
    assert np.array_equal(
        dictionary_learning_minibatch_partial_fit_existing_dictionary(fitted.components_),
        fitted.components_,
    )
    assert np.array_equal(
        dictionary_learning_minibatch_partial_fit_components(fitted.components_),
        fitted.components_,
    )
    assert dictionary_learning_minibatch_partial_fit_updated_n_steps(fitted.n_steps_ - 1) == fitted.n_steps_


def test_dictionary_learning_minibatch_partial_fit_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell import (
        dictionary_learning_minibatch_partial_fit_initial_dictionary,
        dictionary_learning_minibatch_partial_fit_initial_inner_stats,
        dictionary_learning_minibatch_partial_fit_updated_n_steps,
    )

    with pytest.raises(Exception):
        dictionary_learning_minibatch_partial_fit_initial_dictionary(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises(Exception):
        dictionary_learning_minibatch_partial_fit_initial_inner_stats(0, 3, "float64")

    with pytest.raises(Exception):
        dictionary_learning_minibatch_partial_fit_initial_inner_stats(2, 3, "int64")

    with pytest.raises(Exception):
        dictionary_learning_minibatch_partial_fit_updated_n_steps(-1)

    with pytest.raises(Exception):
        dictionary_learning_minibatch_partial_fit_initial_n_steps(False)
