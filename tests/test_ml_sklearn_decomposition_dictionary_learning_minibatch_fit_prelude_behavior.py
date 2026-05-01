from __future__ import annotations

import numpy as np
import pytest


def test_dictionary_learning_minibatch_fit_prelude_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_fit_prelude import (
        dictionary_learning_minibatch_monitor_state,
        dictionary_learning_minibatch_old_dictionary,
        dictionary_learning_minibatch_training_data,
        dictionary_learning_minibatch_verbose_message,
    )

    assert callable(dictionary_learning_minibatch_training_data)
    assert callable(dictionary_learning_minibatch_old_dictionary)
    assert callable(dictionary_learning_minibatch_verbose_message)
    assert callable(dictionary_learning_minibatch_monitor_state)


def test_dictionary_learning_minibatch_fit_prelude_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_fit_prelude import (
        dictionary_learning_minibatch_monitor_state,
        dictionary_learning_minibatch_old_dictionary,
        dictionary_learning_minibatch_training_data,
        dictionary_learning_minibatch_verbose_message,
    )

    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    assert np.array_equal(dictionary_learning_minibatch_training_data(X, False), X)

    permutation = np.array([2, 0, 1], dtype=np.int64)
    assert np.array_equal(
        dictionary_learning_minibatch_training_data(X, True, permutation),
        X[permutation],
    )

    dictionary = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    copied = dictionary_learning_minibatch_old_dictionary(dictionary)
    assert np.array_equal(copied, dictionary)
    assert copied is not dictionary

    assert dictionary_learning_minibatch_verbose_message(False) is None
    assert dictionary_learning_minibatch_verbose_message(True) == "[dict_learning]"
    assert dictionary_learning_minibatch_verbose_message(2) == "[dict_learning]"

    assert dictionary_learning_minibatch_monitor_state(None) == (None, None, 0)
    assert dictionary_learning_minibatch_monitor_state(5) == (None, None, 0)


def test_dictionary_learning_minibatch_fit_prelude_contracts() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_fit_prelude import (
        dictionary_learning_minibatch_training_data,
        dictionary_learning_minibatch_verbose_message,
    )

    X = np.ones((3, 2), dtype=np.float64)

    with pytest.raises(Exception):
        dictionary_learning_minibatch_training_data(X, True, np.array([0, 0, 1], dtype=np.int64))

    with pytest.raises(Exception):
        dictionary_learning_minibatch_training_data(X, False, np.array([0, 1, 2], dtype=np.int64))

    with pytest.raises(Exception):
        dictionary_learning_minibatch_verbose_message(np.array([1]))
