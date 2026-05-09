"""Sklearn coordinate-descent LinearModelCV constraint atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_base_parameter_constraint_descriptors,
    witness_cd_cv_base_parameter_constraint_names,
)


_CONSTRAINT_NAMES = (
    "eps",
    "n_alphas",
    "alphas",
    "fit_intercept",
    "precompute",
    "max_iter",
    "tol",
    "copy_X",
    "cv",
    "verbose",
    "n_jobs",
    "positive",
    "random_state",
    "selection",
)

_CONSTRAINT_DESCRIPTORS = {
    "eps": (("interval", "Real", 0, None, "neither"),),
    "n_alphas": (("interval", "Integral", 1, None, "left"),),
    "alphas": ("array-like", None),
    "fit_intercept": ("boolean",),
    "precompute": (("str_options", ("auto",)), "array-like", "boolean"),
    "max_iter": (("interval", "Integral", 1, None, "left"),),
    "tol": (("interval", "Real", 0, None, "left"),),
    "copy_X": ("boolean",),
    "cv": ("cv_object",),
    "verbose": ("verbose",),
    "n_jobs": ("Integral", None),
    "positive": ("boolean",),
    "random_state": ("random_state",),
    "selection": (("str_options", ("cyclic", "random")),),
}


@register_atom(witness_cd_cv_base_parameter_constraint_names)
@icontract.require(lambda estimator_kind: estimator_kind == "linear_model_cv", "estimator_kind must be linear_model_cv")
@icontract.ensure(
    lambda result: isinstance(result, tuple) and result == _CONSTRAINT_NAMES,
    "LinearModelCV constraint names must preserve sklearn declaration order",
)
def cd_cv_base_parameter_constraint_names(estimator_kind: str) -> tuple[str, ...]:
    """Return LinearModelCV._parameter_constraints names in declaration order."""
    del estimator_kind
    return _CONSTRAINT_NAMES


@register_atom(witness_cd_cv_base_parameter_constraint_descriptors)
@icontract.require(lambda estimator_kind: estimator_kind == "linear_model_cv", "estimator_kind must be linear_model_cv")
@icontract.ensure(
    lambda result: isinstance(result, dict)
    and tuple(result) == _CONSTRAINT_NAMES
    and result == _CONSTRAINT_DESCRIPTORS,
    "LinearModelCV constraint descriptors must match the class declaration",
)
def cd_cv_base_parameter_constraint_descriptors(estimator_kind: str) -> dict[str, tuple[object, ...]]:
    """Return compact descriptors for LinearModelCV._parameter_constraints."""
    del estimator_kind
    return dict(_CONSTRAINT_DESCRIPTORS)
