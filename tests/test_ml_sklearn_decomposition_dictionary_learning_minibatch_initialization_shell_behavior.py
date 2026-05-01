from __future__ import annotations

import numpy as np
import pytest


def test_dictionary_learning_minibatch_initialization_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_initialization_shell import (
        dictionary_learning_minibatch_dictionary_buffer,
        dictionary_learning_minibatch_initial_dictionary,
        dictionary_learning_minibatch_resize_dictionary,
    )

    assert callable(dictionary_learning_minibatch_initial_dictionary)
    assert callable(dictionary_learning_minibatch_resize_dictionary)
    assert callable(dictionary_learning_minibatch_dictionary_buffer)


def test_dictionary_learning_minibatch_initialization_shell_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_initialization_shell import (
        dictionary_learning_minibatch_dictionary_buffer,
        dictionary_learning_minibatch_initial_dictionary,
        dictionary_learning_minibatch_resize_dictionary,
    )

    dict_init = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    svd_dictionary = np.array([[9.0, 8.0]], dtype=np.float64)
    selected = dictionary_learning_minibatch_initial_dictionary(dict_init, None)
    assert np.array_equal(selected, dict_init)
    selected = dictionary_learning_minibatch_initial_dictionary(None, svd_dictionary)
    assert np.array_equal(selected, svd_dictionary)

    resized = dictionary_learning_minibatch_resize_dictionary(dict_init, 1)
    assert np.array_equal(resized, dict_init[:1, :])

    padded = dictionary_learning_minibatch_resize_dictionary(dict_init, 3)
    expected_padded = np.vstack([dict_init, np.zeros((1, 2), dtype=np.float64)])
    assert np.array_equal(padded, expected_padded)

    c_order = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order="C")
    buffered = dictionary_learning_minibatch_dictionary_buffer(c_order, "float32")
    assert np.array_equal(buffered, c_order)
    assert buffered.flags["F_CONTIGUOUS"]
    assert buffered.flags["WRITEABLE"]
    assert buffered.dtype == np.float32


def test_dictionary_learning_minibatch_initialization_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_initialization_shell import (
        dictionary_learning_minibatch_dictionary_buffer,
        dictionary_learning_minibatch_initial_dictionary,
        dictionary_learning_minibatch_resize_dictionary,
    )

    with pytest.raises(Exception):
        dictionary_learning_minibatch_initial_dictionary(None, None)

    with pytest.raises(Exception):
        dictionary_learning_minibatch_initial_dictionary(np.ones((1, 1)), np.ones((1, 1)))

    with pytest.raises(Exception):
        dictionary_learning_minibatch_resize_dictionary(np.ones((1, 1)), 0)

    with pytest.raises(Exception):
        dictionary_learning_minibatch_dictionary_buffer(np.ones((1, 1)), "float16")
