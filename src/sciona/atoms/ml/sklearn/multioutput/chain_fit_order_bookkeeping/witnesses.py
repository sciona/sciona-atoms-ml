"""Ghost witnesses for multioutput chain fit-order bookkeeping helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_chain_fit_tuple_order_array(order: tuple[int, ...]) -> AbstractArray:
    """Describe tuple-configured chain order after sklearn converts it to ndarray."""
    if not isinstance(order, tuple) or len(order) < 1:
        raise ValueError("order must be a nonempty tuple")
    return AbstractArray(shape=(len(order),), dtype="int64")


def witness_chain_fit_require_valid_order(
    order: list[int] | AbstractArray,
    n_outputs: int,
) -> AbstractArray:
    """Describe the Boolean success flag from explicit chain-order validation."""
    del order
    if not isinstance(n_outputs, int) or n_outputs < 1:
        raise ValueError("n_outputs must be positive")
    return AbstractArray(shape=(), dtype="bool")


def witness_chain_fit_log_message(
    verbose: bool,
    estimator_idx: int,
    n_estimators: int,
    processing_msg: str,
) -> AbstractArray:
    """Describe sklearn's optional verbose chain-fit message for one estimator step."""
    del verbose, estimator_idx, n_estimators, processing_msg
    return AbstractArray(shape=(), dtype="object")
