"""Ghost witnesses for MLP fit-shell bookkeeping helper atoms."""

from __future__ import annotations

from collections.abc import Sequence


def witness_mlp_hidden_layer_sizes(
    hidden_layer_sizes: int | Sequence[int],
) -> tuple[int, ...]:
    """Describe normalized hidden-layer sizes before MLP parameter setup."""
    del hidden_layer_sizes
    return (1,)


def witness_mlp_first_pass_required(
    *,
    has_coefs: bool,
    warm_start: bool,
    incremental: bool,
) -> bool:
    """Describe the first-pass decision for fit or partial_fit."""
    del has_coefs
    del warm_start
    del incremental
    return True


def witness_mlp_partial_fit_require_no_early_stopping(
    *,
    early_stopping: bool,
    incremental: bool,
) -> bool:
    """Describe the early-stopping guard in incremental MLP training."""
    del early_stopping
    del incremental
    return True


def witness_mlp_batch_size_warning_required(
    batch_size: int | str,
    *,
    n_samples: int,
) -> bool:
    """Describe whether stochastic MLP fit would trigger the batch-size clipping warning."""
    del batch_size
    del n_samples
    return False


def witness_mlp_batch_size(
    batch_size: int | str,
    *,
    n_samples: int,
) -> int:
    """Describe the resolved stochastic MLP batch size."""
    del batch_size
    del n_samples
    return 1
