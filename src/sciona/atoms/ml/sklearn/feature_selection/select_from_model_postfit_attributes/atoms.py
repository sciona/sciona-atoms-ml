"""SelectFromModel post-fit attribute helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_select_from_model_partial_fit_first_call,
    witness_select_from_model_postfit_feature_names_in,
    witness_select_from_model_postfit_n_features_in,
)


def _bool_flag(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _feature_names_valid(feature_names: object) -> bool:
    return bool(
        isinstance(feature_names, tuple)
        and len(feature_names) >= 1
        and all(isinstance(name, str) and name != "" for name in feature_names)
    )


@register_atom(witness_select_from_model_partial_fit_first_call)
@icontract.require(
    lambda estimator_is_initialized: _bool_flag(estimator_is_initialized),
    "estimator_is_initialized must be boolean",
)
@icontract.ensure(lambda result: _bool_flag(result), "result must be boolean")
def select_from_model_partial_fit_first_call(
    *,
    estimator_is_initialized: bool,
) -> bool:
    """Return sklearn's first-call flag for SelectFromModel.partial_fit."""
    return not estimator_is_initialized


@register_atom(witness_select_from_model_postfit_n_features_in)
@icontract.require(
    lambda n_features_in: _positive_int(n_features_in),
    "n_features_in must be a positive integer",
)
@icontract.ensure(
    lambda result: _positive_int(result),
    "result must be a positive integer",
)
def select_from_model_postfit_n_features_in(n_features_in: int) -> int:
    """Expose sklearn's fitted n_features_in_ copied from the wrapped estimator."""
    return int(n_features_in)


@register_atom(witness_select_from_model_postfit_feature_names_in)
@icontract.require(
    lambda feature_names_in: _feature_names_valid(feature_names_in),
    "feature_names_in must be a nonempty tuple of nonempty strings",
)
@icontract.ensure(
    lambda result: _feature_names_valid(result),
    "result must preserve a nonempty tuple of feature names",
)
def select_from_model_postfit_feature_names_in(
    feature_names_in: tuple[str, ...],
) -> tuple[str, ...]:
    """Expose sklearn's fitted feature_names_in_ copied from the wrapped estimator."""
    return tuple(feature_names_in)
