"""Sklearn coordinate-descent CV alpha-bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_alpha_grid_required,
    witness_cd_cv_default_l1_ratios,
    witness_cd_cv_first_path_l1_ratio,
    witness_cd_cv_has_l1_ratio_param,
    witness_cd_cv_l1_ratios,
    witness_cd_cv_n_alphas,
    witness_cd_cv_n_l1_ratio,
    witness_cd_cv_sorted_alphas,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


@register_atom(witness_cd_cv_has_l1_ratio_param)
@icontract.require(lambda path_params: isinstance(path_params, dict), "path_params must be a dict")
@icontract.ensure(
    lambda result, path_params: _bool(result) and result == ("l1_ratio" in path_params),
    "l1_ratio branch must match sklearn membership testing",
)
def cd_cv_has_l1_ratio_param(path_params: dict[object, object]) -> bool:
    """Return whether LinearModelCV.fit should use the l1_ratio branch."""
    return "l1_ratio" in path_params


@register_atom(witness_cd_cv_l1_ratios)
@icontract.require(lambda l1_ratio_value: l1_ratio_value is not None, "l1_ratio_value must be provided")
@icontract.ensure(
    lambda result: isinstance(result, np.ndarray) and result.ndim >= 1,
    "l1_ratios must be an ndarray with at least one dimension",
)
def cd_cv_l1_ratios(l1_ratio_value: object) -> np.ndarray:
    """Return the l1_ratio vector assembled by np.atleast_1d."""
    return np.atleast_1d(l1_ratio_value)


@register_atom(witness_cd_cv_first_path_l1_ratio)
@icontract.require(
    lambda l1_ratios: isinstance(l1_ratios, np.ndarray) and l1_ratios.size >= 1,
    "l1_ratios must be a nonempty ndarray",
)
@icontract.ensure(
    lambda result, l1_ratios: np.array_equal(np.asarray(result), np.asarray(l1_ratios[0])),
    "first path l1_ratio must equal l1_ratios[0]",
)
def cd_cv_first_path_l1_ratio(l1_ratios: np.ndarray) -> object:
    """Return the l1_ratio value used for the first path call."""
    return l1_ratios[0]


@register_atom(witness_cd_cv_default_l1_ratios)
@icontract.require(
    lambda has_l1_ratio_param: _bool(has_l1_ratio_param),
    "has_l1_ratio_param must be boolean",
)
@icontract.ensure(
    lambda result, has_l1_ratio_param: isinstance(result, list)
    and result == [1]
    and has_l1_ratio_param is False,
    "default l1_ratios must be [1] when the parameter is absent",
)
def cd_cv_default_l1_ratios(has_l1_ratio_param: bool) -> list[int]:
    """Return the default l1_ratio list used when no l1_ratio parameter exists."""
    del has_l1_ratio_param
    return [1]


@register_atom(witness_cd_cv_alpha_grid_required)
@icontract.require(lambda alphas_is_none: _bool(alphas_is_none), "alphas_is_none must be boolean")
@icontract.ensure(
    lambda result, alphas_is_none: _bool(result) and result == alphas_is_none,
    "alpha-grid branch must match alphas is None",
)
def cd_cv_alpha_grid_required(alphas_is_none: bool) -> bool:
    """Return whether LinearModelCV.fit should compute alpha grids internally."""
    return alphas_is_none


@register_atom(witness_cd_cv_sorted_alphas)
@icontract.require(
    lambda alphas: isinstance(alphas, np.ndarray) and alphas.ndim == 1 and alphas.size >= 1,
    "alphas must be a nonempty 1D ndarray",
)
@icontract.require(lambda n_l1_ratio: _positive_int(n_l1_ratio), "n_l1_ratio must be positive")
@icontract.ensure(
    lambda result, alphas, n_l1_ratio: isinstance(result, np.ndarray)
    and result.shape == (int(n_l1_ratio), len(alphas))
    and np.all(result[:, :-1] >= result[:, 1:])
    and np.array_equal(result[0], np.sort(alphas)[::-1]),
    "sorted alphas must be descending and tiled across l1_ratio values",
)
def cd_cv_sorted_alphas(alphas: np.ndarray, n_l1_ratio: int) -> np.ndarray:
    """Return the descending alpha grid tiled for each l1_ratio."""
    return np.tile(np.sort(alphas)[::-1], (int(n_l1_ratio), 1))


@register_atom(witness_cd_cv_n_l1_ratio)
@icontract.require(
    lambda l1_ratios: hasattr(l1_ratios, "__len__"),
    "l1_ratios must have a length",
)
@icontract.ensure(
    lambda result, l1_ratios: _positive_int(result) and int(result) == len(l1_ratios),
    "n_l1_ratio must equal len(l1_ratios)",
)
def cd_cv_n_l1_ratio(l1_ratios: object) -> int:
    """Return the number of l1_ratio values used by LinearModelCV.fit."""
    return len(l1_ratios)


@register_atom(witness_cd_cv_n_alphas)
@icontract.require(
    lambda alphas: hasattr(alphas, "__getitem__") and len(alphas) >= 1,
    "alphas must contain at least one grid",
)
@icontract.ensure(
    lambda result, alphas: _positive_int(result) and int(result) == len(alphas[0]),
    "n_alphas must equal len(alphas[0])",
)
def cd_cv_n_alphas(alphas: object) -> int:
    """Return the alpha count used for each l1_ratio in LinearModelCV.fit."""
    return len(alphas[0])
