"""Gaussian-process classification predict_proba shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_predict_proba_dtype_name,
    witness_gpc_predict_proba_require_supported_multiclass_mode,
    witness_gpc_predict_proba_validate_ensure_2d,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _one_vs_mode(value: object) -> bool:
    return bool(isinstance(value, str) and value in {"one_vs_rest", "one_vs_one"})


@register_atom(witness_gpc_predict_proba_require_supported_multiclass_mode)
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
@icontract.require(lambda multi_class: _one_vs_mode(multi_class), "multi_class must be 'one_vs_rest' or 'one_vs_one'")
@icontract.ensure(lambda result: _bool(result) and result is True, "result must be True when predict_proba is supported for the fitted multiclass mode")
def gpc_predict_proba_require_supported_multiclass_mode(
    n_classes: int,
    multi_class: str,
) -> bool:
    """Apply GaussianProcessClassifier.predict_proba's one-vs-one probability guard."""
    if int(n_classes) > 2 and multi_class == "one_vs_one":
        raise ValueError(
            "one_vs_one multi-class mode does not support "
            "predicting probability estimates. Use "
            "one_vs_rest mode instead."
        )
    return True


@register_atom(witness_gpc_predict_proba_dtype_name)
@icontract.require(
    lambda kernel_is_none_or_requires_vector_input: _bool(kernel_is_none_or_requires_vector_input),
    "kernel_is_none_or_requires_vector_input must be boolean",
)
@icontract.ensure(
    lambda result: result in {None, "numeric"},
    "dtype mode must match sklearn's predict_proba validation choices",
)
def gpc_predict_proba_dtype_name(
    kernel_is_none_or_requires_vector_input: bool,
) -> str | None:
    """Resolve sklearn's predict_proba validate_data dtype mode for Gaussian-process classification."""
    if kernel_is_none_or_requires_vector_input:
        return "numeric"
    return None


@register_atom(witness_gpc_predict_proba_validate_ensure_2d)
@icontract.require(
    lambda kernel_is_none_or_requires_vector_input: _bool(kernel_is_none_or_requires_vector_input),
    "kernel_is_none_or_requires_vector_input must be boolean",
)
@icontract.ensure(lambda result: _bool(result), "ensure_2d mode must be boolean")
def gpc_predict_proba_validate_ensure_2d(
    kernel_is_none_or_requires_vector_input: bool,
) -> bool:
    """Resolve sklearn's predict_proba validate_data ensure_2d mode for Gaussian-process classification."""
    return bool(kernel_is_none_or_requires_vector_input)
