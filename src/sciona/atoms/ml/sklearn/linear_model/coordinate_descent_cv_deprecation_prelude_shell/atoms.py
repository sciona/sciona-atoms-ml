"""Sklearn coordinate-descent CV deprecation-prelude shell atoms adapted from scikit-learn."""

from __future__ import annotations

from numbers import Integral

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_alphas_none_deprecation_message,
    witness_cd_cv_alphas_none_deprecation_warning_required,
    witness_cd_cv_alphas_warn_sentinel,
    witness_cd_cv_n_alphas_deprecation_message,
    witness_cd_cv_n_alphas_deprecation_warning_required,
    witness_cd_cv_resolved_alphas,
)


_N_ALPHAS_DEPRECATED_MESSAGE = (
    "'n_alphas' was deprecated in 1.7 and will be removed in 1.9. "
    "'alphas' now accepts an integer value which removes the need to pass "
    "'n_alphas'. The default value of 'alphas' will change from None to "
    "100 in 1.9. Pass an explicit value to 'alphas' and leave 'n_alphas' "
    "to its default value to silence this warning."
)

_ALPHAS_NONE_DEPRECATED_MESSAGE = (
    "'alphas=None' is deprecated and will be removed in 1.9, at which "
    "point the default value will be set to 100. Set 'alphas=100' "
    "to silence this warning."
)


def _positive_integral(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 1


def _warn_sentinel(value: object) -> bool:
    return isinstance(value, str) and value == "warn"


@register_atom(witness_cd_cv_n_alphas_deprecation_warning_required)
@icontract.require(
    lambda n_alphas: n_alphas == "deprecated" or _positive_integral(n_alphas),
    "n_alphas must be the deprecated sentinel or a positive integer",
)
@icontract.ensure(
    lambda result, n_alphas: isinstance(result, bool)
    and result == (n_alphas != "deprecated"),
    "n_alphas warning predicate must match sklearn's deprecated sentinel branch",
)
def cd_cv_n_alphas_deprecation_warning_required(n_alphas: object) -> bool:
    """Return whether LinearModelCV.fit warns for explicit n_alphas."""
    return n_alphas != "deprecated"


@register_atom(witness_cd_cv_n_alphas_deprecation_message)
@icontract.require(lambda warning_required: warning_required is True, "warning_required must be true")
@icontract.ensure(
    lambda result: result == _N_ALPHAS_DEPRECATED_MESSAGE,
    "n_alphas deprecation message must match sklearn 1.8.0",
)
def cd_cv_n_alphas_deprecation_message(warning_required: bool) -> str:
    """Return the FutureWarning text for deprecated n_alphas usage."""
    del warning_required
    return _N_ALPHAS_DEPRECATED_MESSAGE


@register_atom(witness_cd_cv_alphas_warn_sentinel)
@icontract.ensure(
    lambda result, alphas: isinstance(result, bool) and result == _warn_sentinel(alphas),
    "alphas warn sentinel predicate must match sklearn's string sentinel guard",
)
def cd_cv_alphas_warn_sentinel(alphas: object) -> bool:
    """Return whether alphas is the sklearn 1.8.0 internal 'warn' sentinel."""
    return _warn_sentinel(alphas)


@register_atom(witness_cd_cv_alphas_none_deprecation_warning_required)
@icontract.ensure(
    lambda result, alphas: isinstance(result, bool) and result == (alphas is None),
    "alphas=None warning predicate must match sklearn's deprecation branch",
)
def cd_cv_alphas_none_deprecation_warning_required(alphas: object) -> bool:
    """Return whether LinearModelCV.fit warns for alphas=None."""
    return alphas is None


@register_atom(witness_cd_cv_alphas_none_deprecation_message)
@icontract.require(lambda warning_required: warning_required is True, "warning_required must be true")
@icontract.ensure(
    lambda result: result == _ALPHAS_NONE_DEPRECATED_MESSAGE,
    "alphas=None deprecation message must match sklearn 1.8.0",
)
def cd_cv_alphas_none_deprecation_message(warning_required: bool) -> str:
    """Return the FutureWarning text for deprecated alphas=None usage."""
    del warning_required
    return _ALPHAS_NONE_DEPRECATED_MESSAGE


@register_atom(witness_cd_cv_resolved_alphas)
@icontract.require(
    lambda n_alphas: n_alphas == "deprecated" or _positive_integral(n_alphas),
    "n_alphas must be the deprecated sentinel or a positive integer",
)
@icontract.ensure(
    lambda result, n_alphas, alphas: (
        (result == 100)
        if (n_alphas == "deprecated" and (_warn_sentinel(alphas) or alphas is None))
        else (
            (result is n_alphas)
            if (n_alphas != "deprecated" and (_warn_sentinel(alphas) or alphas is None))
            else (result is alphas)
        )
    ),
    "_alphas resolution must follow LinearModelCV.fit deprecation prelude",
)
def cd_cv_resolved_alphas(n_alphas: object, alphas: object) -> object:
    """Return the private _alphas value after the deprecation prelude."""
    resolved: object = 100 if n_alphas == "deprecated" else n_alphas
    if _warn_sentinel(alphas):
        return resolved
    if alphas is None:
        return resolved
    return alphas
