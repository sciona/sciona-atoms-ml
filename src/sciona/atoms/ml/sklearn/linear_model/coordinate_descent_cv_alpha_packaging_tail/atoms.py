"""Sklearn coordinate-descent CV alpha-packaging tail atoms."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_auto_alphas_array,
    witness_cd_cv_auto_alphas_packaging_required,
    witness_cd_cv_auto_alphas_public,
    witness_cd_cv_auto_alphas_single_ratio_collapse_required,
    witness_cd_cv_user_alphas_packaging_required,
    witness_cd_cv_user_alphas_public,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


@register_atom(witness_cd_cv_auto_alphas_packaging_required)
@icontract.require(lambda self_alphas_is_none: _bool(self_alphas_is_none), "self_alphas_is_none must be boolean")
@icontract.ensure(
    lambda result, self_alphas_is_none: _bool(result) and result == self_alphas_is_none,
    "auto alpha packaging branch must match self.alphas is None",
)
def cd_cv_auto_alphas_packaging_required(self_alphas_is_none: bool) -> bool:
    """Return whether LinearModelCV.fit packages computed alpha grids."""
    return self_alphas_is_none


@register_atom(witness_cd_cv_user_alphas_packaging_required)
@icontract.require(lambda self_alphas_is_none: _bool(self_alphas_is_none), "self_alphas_is_none must be boolean")
@icontract.ensure(
    lambda result, self_alphas_is_none: _bool(result) and result == (not self_alphas_is_none),
    "user alpha packaging branch must match self.alphas is not None",
)
def cd_cv_user_alphas_packaging_required(self_alphas_is_none: bool) -> bool:
    """Return whether LinearModelCV.fit packages a user-provided alpha grid."""
    return not self_alphas_is_none


@register_atom(witness_cd_cv_auto_alphas_array)
@icontract.require(lambda auto_packaging_required: auto_packaging_required is True, "auto packaging must be required")
@icontract.require(
    lambda alphas: hasattr(alphas, "__len__") and len(alphas) >= 1,
    "alphas must contain at least one computed grid",
)
@icontract.ensure(
    lambda result: isinstance(result, np.ndarray) and result.shape[0] >= 1,
    "auto alpha array must be a nonempty ndarray",
)
def cd_cv_auto_alphas_array(alphas: object, auto_packaging_required: bool) -> np.ndarray:
    """Return np.asarray(alphas) for computed alpha grids."""
    del auto_packaging_required
    return np.asarray(alphas)


@register_atom(witness_cd_cv_auto_alphas_single_ratio_collapse_required)
@icontract.require(lambda n_l1_ratio: _positive_int(n_l1_ratio), "n_l1_ratio must be positive")
@icontract.ensure(
    lambda result, n_l1_ratio: _bool(result) and result == (int(n_l1_ratio) == 1),
    "single-ratio collapse branch must match n_l1_ratio == 1",
)
def cd_cv_auto_alphas_single_ratio_collapse_required(n_l1_ratio: int) -> bool:
    """Return whether LinearModelCV.fit collapses computed alphas_[0]."""
    return int(n_l1_ratio) == 1


@register_atom(witness_cd_cv_auto_alphas_public)
@icontract.require(
    lambda auto_alphas_array: isinstance(auto_alphas_array, np.ndarray)
    and auto_alphas_array.shape[0] >= 1,
    "auto_alphas_array must be a nonempty ndarray",
)
@icontract.require(lambda collapse_required: _bool(collapse_required), "collapse_required must be boolean")
@icontract.ensure(
    lambda result, auto_alphas_array, collapse_required: isinstance(result, np.ndarray)
    and (
        np.array_equal(result, auto_alphas_array[0])
        if collapse_required
        else np.array_equal(result, auto_alphas_array)
    ),
    "public computed alphas_ must match sklearn collapse branch",
)
def cd_cv_auto_alphas_public(
    auto_alphas_array: np.ndarray, collapse_required: bool
) -> np.ndarray:
    """Return public alphas_ packaging for computed alpha grids."""
    if collapse_required:
        return auto_alphas_array[0]
    return auto_alphas_array


@register_atom(witness_cd_cv_user_alphas_public)
@icontract.require(lambda user_packaging_required: user_packaging_required is True, "user packaging must be required")
@icontract.require(
    lambda alphas: hasattr(alphas, "__getitem__") and len(alphas) >= 1,
    "alphas must contain at least one user grid",
)
@icontract.ensure(
    lambda result, alphas: isinstance(result, np.ndarray)
    and np.array_equal(result, np.asarray(alphas[0])),
    "public user alphas_ must equal np.asarray(alphas[0])",
)
def cd_cv_user_alphas_public(alphas: object, user_packaging_required: bool) -> np.ndarray:
    """Return public alphas_ packaging for user-provided alpha grids."""
    del user_packaging_required
    return np.asarray(alphas[0])
