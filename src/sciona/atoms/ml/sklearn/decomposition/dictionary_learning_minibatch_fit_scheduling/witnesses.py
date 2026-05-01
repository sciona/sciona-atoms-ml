"""Ghost witnesses for MiniBatchDictionaryLearning fit-scheduling atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_dictionary_learning_minibatch_inner_stat_buffers(
    n_features: int,
    n_components: int,
    dtype_name: str,
):
    """Describe the zero-initialized inner-stat buffers."""
    del dtype_name
    return (
        AbstractArray(shape=(n_components, n_components), dtype="float64"),
        AbstractArray(shape=(n_features, n_components), dtype="float64"),
    )


def witness_dictionary_learning_minibatch_steps_per_iter(
    n_samples: int,
    batch_size: int,
) -> int:
    """Describe the number of minibatches per outer iteration."""
    del n_samples, batch_size
    return 1


def witness_dictionary_learning_minibatch_total_steps(
    max_iter: int,
    steps_per_iter: int,
) -> int:
    """Describe the total number of minibatch loop steps."""
    del max_iter, steps_per_iter
    return 1


def witness_dictionary_learning_minibatch_fit_counters(
    final_step_index: int,
    steps_per_iter: int,
):
    """Describe the final n_steps_ and n_iter_ counters."""
    del final_step_index, steps_per_iter
    return (
        AbstractArray(shape=(), dtype="int64"),
        AbstractArray(shape=(), dtype="float64"),
    )
