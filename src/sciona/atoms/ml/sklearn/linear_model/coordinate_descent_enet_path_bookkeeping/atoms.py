"""Sklearn coordinate-descent enet_path bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_enet_path_multi_output,
    witness_cd_enet_path_outputs,
    witness_cd_enet_path_positive_multi_output_guard_required,
    witness_cd_enet_path_random_selection,
    witness_cd_enet_path_regularization_pair,
    witness_cd_enet_path_sorted_alphas,
    witness_cd_enet_path_target_count,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_tensor(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim >= 1 and array.size >= 1 and np.all(np.isfinite(array)))


@register_atom(witness_cd_enet_path_multi_output)
@icontract.require(lambda y_ndim: _positive_int(y_ndim), "y_ndim must be a positive integer")
@icontract.ensure(
    lambda result, y_ndim: isinstance(result, bool) and result == (int(y_ndim) != 1),
    "multi_output must be true exactly when y.ndim != 1",
)
def cd_enet_path_multi_output(y_ndim: int) -> bool:
    """Return whether enet_path is in multi-output mode."""
    return int(y_ndim) != 1


@register_atom(witness_cd_enet_path_target_count)
@icontract.require(
    lambda y_shape: isinstance(y_shape, tuple) and len(y_shape) >= 1 and all(_positive_int(v) for v in y_shape),
    "y_shape must be a tuple of positive integers",
)
@icontract.require(lambda multi_output: isinstance(multi_output, bool), "multi_output must be boolean")
@icontract.ensure(
    lambda result, y_shape, multi_output: (result is None and not multi_output)
    or (_positive_int(result) and multi_output and int(result) == int(y_shape[1])),
    "target_count must be y.shape[1] in multi-output mode and None otherwise",
)
def cd_enet_path_target_count(y_shape: tuple[int, ...], multi_output: bool) -> int | None:
    """Return the target count used by enet_path in multi-output mode."""
    if not multi_output:
        return None
    return int(y_shape[1])


@register_atom(witness_cd_enet_path_positive_multi_output_guard_required)
@icontract.require(lambda multi_output: isinstance(multi_output, bool), "multi_output must be boolean")
@icontract.require(lambda positive: isinstance(positive, bool), "positive must be boolean")
@icontract.ensure(
    lambda result, multi_output, positive: isinstance(result, bool) and result == (multi_output and positive),
    "guard predicate must match multi_output and positive",
)
def cd_enet_path_positive_multi_output_guard_required(multi_output: bool, positive: bool) -> bool:
    """Return whether enet_path should raise on positive multi-output input."""
    return multi_output and positive


@register_atom(witness_cd_enet_path_sorted_alphas)
@icontract.require(lambda alphas: _finite_vector(alphas), "alphas must be a finite one-dimensional vector")
@icontract.ensure(
    lambda result, alphas: _finite_vector(result)
    and np.asarray(result, dtype=np.float64).shape == np.asarray(alphas, dtype=np.float64).shape
    and (
        np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(alphas, dtype=np.float64))
        if np.asarray(alphas, dtype=np.float64).shape[0] <= 1
        else np.array_equal(
            np.asarray(result, dtype=np.float64),
            np.sort(np.asarray(alphas, dtype=np.float64))[::-1],
        )
    ),
    "alphas must be preserved when length <= 1, otherwise sorted descending",
)
def cd_enet_path_sorted_alphas(alphas: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return the alpha vector after enet_path's descending sort shell."""
    values = np.asarray(alphas, dtype=np.float64)
    if values.shape[0] > 1:
        values = np.sort(values)[::-1]
    return np.asarray(values, dtype=np.float64)


@register_atom(witness_cd_enet_path_random_selection)
@icontract.require(
    lambda selection: isinstance(selection, str) and selection in {"random", "cyclic"},
    "selection must be either 'random' or 'cyclic'",
)
@icontract.ensure(
    lambda result, selection: isinstance(result, bool) and result == (selection == "random"),
    "random_selection must match selection == 'random'",
)
def cd_enet_path_random_selection(selection: str) -> bool:
    """Return whether enet_path should use random coordinate selection."""
    return selection == "random"


@register_atom(witness_cd_enet_path_regularization_pair)
@icontract.require(lambda alpha: np.isfinite(float(alpha)), "alpha must be finite")
@icontract.require(lambda l1_ratio: np.isfinite(float(l1_ratio)), "l1_ratio must be finite")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(
    lambda result, alpha, l1_ratio, n_samples: isinstance(result, tuple)
    and len(result) == 2
    and np.allclose(
        np.asarray(result, dtype=np.float64),
        np.array(
            [
                float(alpha) * float(l1_ratio) * int(n_samples),
                float(alpha) * (1.0 - float(l1_ratio)) * int(n_samples),
            ],
            dtype=np.float64,
        ),
    ),
    "regularization pair must match the l1/l2 scaling shell",
)
def cd_enet_path_regularization_pair(alpha: float, l1_ratio: float, n_samples: int) -> tuple[float, float]:
    """Return the scaled l1_reg and l2_reg pair used by enet_path."""
    return (
        float(alpha) * float(l1_ratio) * int(n_samples),
        float(alpha) * (1.0 - float(l1_ratio)) * int(n_samples),
    )


@register_atom(witness_cd_enet_path_outputs)
@icontract.require(lambda alphas: _finite_vector(alphas), "alphas must be a finite one-dimensional vector")
@icontract.require(lambda coefs: _finite_tensor(coefs), "coefs must be a finite numeric array")
@icontract.require(lambda dual_gaps: _finite_vector(dual_gaps), "dual_gaps must be a finite one-dimensional vector")
@icontract.require(
    lambda n_iters: isinstance(n_iters, Sequence) and len(n_iters) >= 1 and all(_positive_int(v) for v in n_iters),
    "n_iters must be a nonempty sequence of positive integers",
)
@icontract.require(lambda return_n_iter: isinstance(return_n_iter, bool), "return_n_iter must be boolean")
@icontract.ensure(
    lambda result, return_n_iter: isinstance(result, tuple) and len(result) == (4 if return_n_iter else 3),
    "output tuple must include n_iters only when return_n_iter is true",
)
def cd_enet_path_outputs(
    alphas: NDArray[np.floating],
    coefs: NDArray[np.floating],
    dual_gaps: NDArray[np.floating],
    n_iters: Sequence[int],
    return_n_iter: bool,
) -> tuple[object, ...]:
    """Return the final output tuple packaged by enet_path."""
    if return_n_iter:
        return (
            np.asarray(alphas, dtype=np.float64),
            np.asarray(coefs),
            np.asarray(dual_gaps, dtype=np.float64),
            list(int(v) for v in n_iters),
        )
    return (
        np.asarray(alphas, dtype=np.float64),
        np.asarray(coefs),
        np.asarray(dual_gaps, dtype=np.float64),
    )
