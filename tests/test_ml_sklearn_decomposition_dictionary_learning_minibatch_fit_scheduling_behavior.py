from __future__ import annotations

import numpy as np
import pytest


def test_dictionary_learning_minibatch_fit_scheduling_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_fit_scheduling import (
        dictionary_learning_minibatch_fit_counters,
        dictionary_learning_minibatch_inner_stat_buffers,
        dictionary_learning_minibatch_steps_per_iter,
        dictionary_learning_minibatch_total_steps,
    )

    assert callable(dictionary_learning_minibatch_inner_stat_buffers)
    assert callable(dictionary_learning_minibatch_steps_per_iter)
    assert callable(dictionary_learning_minibatch_total_steps)
    assert callable(dictionary_learning_minibatch_fit_counters)


def test_dictionary_learning_minibatch_fit_scheduling_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_fit_scheduling import (
        dictionary_learning_minibatch_fit_counters,
        dictionary_learning_minibatch_inner_stat_buffers,
        dictionary_learning_minibatch_steps_per_iter,
        dictionary_learning_minibatch_total_steps,
    )

    A, B = dictionary_learning_minibatch_inner_stat_buffers(5, 3, "float32")
    assert A.shape == (3, 3)
    assert B.shape == (5, 3)
    assert A.dtype == np.float32
    assert B.dtype == np.float32
    assert np.all(A == 0)
    assert np.all(B == 0)

    steps_per_iter = dictionary_learning_minibatch_steps_per_iter(10, 4)
    assert steps_per_iter == 3

    total_steps = dictionary_learning_minibatch_total_steps(7, steps_per_iter)
    assert total_steps == 21

    n_steps, n_iter = dictionary_learning_minibatch_fit_counters(20, steps_per_iter)
    assert n_steps == 21
    assert n_iter == pytest.approx(7.0)


def test_dictionary_learning_minibatch_fit_scheduling_contracts() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_fit_scheduling import (
        dictionary_learning_minibatch_fit_counters,
        dictionary_learning_minibatch_inner_stat_buffers,
        dictionary_learning_minibatch_steps_per_iter,
        dictionary_learning_minibatch_total_steps,
    )

    with pytest.raises(Exception):
        dictionary_learning_minibatch_inner_stat_buffers(3, 2, "float16")

    with pytest.raises(Exception):
        dictionary_learning_minibatch_steps_per_iter(0, 2)

    with pytest.raises(Exception):
        dictionary_learning_minibatch_total_steps(-1, 2)

    with pytest.raises(Exception):
        dictionary_learning_minibatch_fit_counters(-1, 2)
