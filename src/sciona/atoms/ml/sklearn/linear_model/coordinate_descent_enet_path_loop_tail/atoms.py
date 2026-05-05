"""Sklearn coordinate-descent enet_path loop-tail atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_enet_path_model_coef,
    witness_cd_enet_path_model_iteration_count,
    witness_cd_enet_path_scaled_dual_gap,
    witness_cd_enet_path_selection_error_message,
    witness_cd_enet_path_selection_guard_required,
    witness_cd_enet_path_verbose_progress_message,
    witness_cd_enet_path_verbose_use_progress_print,
    witness_cd_enet_path_verbose_use_stderr_dot,
    witness_cd_enet_path_verbose_use_tuple_print,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _valid_model_tuple(model: object) -> bool:
    return isinstance(model, tuple) and len(model) == 4


@register_atom(witness_cd_enet_path_selection_guard_required)
@icontract.ensure(
    lambda result, selection: isinstance(result, bool)
    and result == (selection not in ["random", "cyclic"]),
    "selection guard must match the sklearn membership test",
)
def cd_enet_path_selection_guard_required(selection: object) -> bool:
    """Return whether enet_path should raise on an invalid selection mode."""
    return selection not in ["random", "cyclic"]


@register_atom(witness_cd_enet_path_selection_error_message)
@icontract.ensure(
    lambda result, selection: isinstance(result, str)
    and result == "selection should be either random or cyclic.",
    "selection error message must match sklearn formatting",
)
def cd_enet_path_selection_error_message(selection: object) -> str:
    """Return the invalid-selection ValueError text used by enet_path."""
    del selection
    return "selection should be either random or cyclic."


@register_atom(witness_cd_enet_path_model_coef)
@icontract.require(lambda model: _valid_model_tuple(model), "model must be a four-item solver tuple")
@icontract.ensure(
    lambda result, model: np.array_equal(np.asarray(result), np.asarray(model[0])),
    "coefficient extraction must return model[0]",
)
def cd_enet_path_model_coef(model: tuple[object, object, object, object]) -> NDArray[np.generic]:
    """Return the coefficient array extracted from a solver result tuple."""
    return np.asarray(model[0])


@register_atom(witness_cd_enet_path_scaled_dual_gap)
@icontract.require(lambda dual_gap: np.isfinite(float(dual_gap)), "dual_gap must be finite")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be positive")
@icontract.ensure(
    lambda result, dual_gap, n_samples: np.isfinite(float(result))
    and np.isclose(float(result), float(dual_gap) / int(n_samples)),
    "scaled dual gap must equal dual_gap / n_samples",
)
def cd_enet_path_scaled_dual_gap(dual_gap: float, n_samples: int) -> float:
    """Return the dual gap scaled back to the public objective."""
    return float(dual_gap) / int(n_samples)


@register_atom(witness_cd_enet_path_model_iteration_count)
@icontract.require(lambda model: _valid_model_tuple(model), "model must be a four-item solver tuple")
@icontract.ensure(
    lambda result, model: _positive_int(result) and int(result) == int(model[3]),
    "iteration-count extraction must return model[3]",
)
def cd_enet_path_model_iteration_count(model: tuple[object, object, object, object]) -> int:
    """Return the iteration count extracted from a solver result tuple."""
    return int(model[3])


@register_atom(witness_cd_enet_path_verbose_use_tuple_print)
@icontract.require(lambda verbose: isinstance(verbose, (bool, int, np.integer)), "verbose must be bool or integer")
@icontract.ensure(
    lambda result, verbose: isinstance(result, bool) and result == (int(verbose) > 2),
    "tuple-print predicate must match verbose > 2",
)
def cd_enet_path_verbose_use_tuple_print(verbose: bool | int) -> bool:
    """Return whether enet_path should print the full solver tuple."""
    return int(verbose) > 2


@register_atom(witness_cd_enet_path_verbose_use_progress_print)
@icontract.require(lambda verbose: isinstance(verbose, (bool, int, np.integer)), "verbose must be bool or integer")
@icontract.ensure(
    lambda result, verbose: isinstance(result, bool) and result == (int(verbose) > 1),
    "progress-print predicate must match verbose > 1",
)
def cd_enet_path_verbose_use_progress_print(verbose: bool | int) -> bool:
    """Return whether enet_path should print the progress message."""
    return int(verbose) > 1


@register_atom(witness_cd_enet_path_verbose_use_stderr_dot)
@icontract.require(lambda verbose: isinstance(verbose, (bool, int, np.integer)), "verbose must be bool or integer")
@icontract.ensure(
    lambda result, verbose: isinstance(result, bool) and result == (bool(verbose) and int(verbose) <= 1),
    "stderr-dot predicate must match the fallback verbose branch",
)
def cd_enet_path_verbose_use_stderr_dot(verbose: bool | int) -> bool:
    """Return whether enet_path should emit the stderr dot."""
    return bool(verbose) and int(verbose) <= 1


@register_atom(witness_cd_enet_path_verbose_progress_message)
@icontract.require(lambda index: isinstance(index, (int, np.integer)) and int(index) >= 0, "index must be nonnegative")
@icontract.require(lambda alpha_count: _positive_int(alpha_count), "alpha_count must be positive")
@icontract.ensure(
    lambda result, index, alpha_count: isinstance(result, str)
    and result == ("Path: %03i out of %03i" % (int(index), int(alpha_count))),
    "progress message must match sklearn formatting",
)
def cd_enet_path_verbose_progress_message(index: int, alpha_count: int) -> str:
    """Return the verbose progress message used by enet_path."""
    return "Path: %03i out of %03i" % (int(index), int(alpha_count))
