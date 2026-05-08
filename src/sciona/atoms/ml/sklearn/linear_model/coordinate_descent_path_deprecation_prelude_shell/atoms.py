"""Sklearn coordinate-descent path deprecation-prelude atoms adapted from scikit-learn."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_path_alphas_none_warning_message,
    witness_cd_path_alphas_none_warning_required,
    witness_cd_path_default_n_alphas_resolution,
    witness_cd_path_effective_alphas_resolution,
    witness_cd_path_n_alphas_warning_message,
    witness_cd_path_n_alphas_warning_required,
)


_N_ALPHAS_DEPRECATED_MESSAGE = (
    "'n_alphas' was deprecated in 1.9 and will be removed in 1.11. "
    "'alphas' now accepts an integer value which removes the need to pass "
    "'n_alphas'. The default value of 'alphas' will change from None to "
    "100 in 1.11. Pass an explicit value to 'alphas' and leave 'n_alphas' "
    "to its default value to silence this warning."
)

_ALPHAS_NONE_DEPRECATED_MESSAGE = (
    "'alphas=None' is deprecated and will be removed in 1.11, at which "
    "point the default value will be set to 100. Set 'alphas=100' "
    "to silence this warning."
)


def _positive_integral(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 1


def _n_alphas_valid(value: object) -> bool:
    return value == "deprecated" or _positive_integral(value)


def _warn_sentinel(value: object) -> bool:
    return isinstance(value, str) and value == "warn"


def _path_helper_name(value: object) -> bool:
    return value in {"lasso_path", "enet_path"}


def _valid_alphas_value(value: object) -> bool:
    return value is None or _warn_sentinel(value) or not isinstance(value, str)


@register_atom(witness_cd_path_default_n_alphas_resolution)
@icontract.require(lambda n_alphas: _n_alphas_valid(n_alphas), "n_alphas must be deprecated or positive")
@icontract.ensure(
    lambda result, n_alphas: _positive_integral(result)
    and int(result) == (100 if n_alphas == "deprecated" else int(n_alphas)),
    "effective n_alphas must match sklearn's sentinel branch",
)
def cd_path_default_n_alphas_resolution(n_alphas: object) -> int:
    """Resolve the effective alpha count before path deprecation handling."""
    if n_alphas == "deprecated":
        return 100
    return int(n_alphas)


@register_atom(witness_cd_path_n_alphas_warning_required)
@icontract.require(lambda n_alphas: _n_alphas_valid(n_alphas), "n_alphas must be deprecated or positive")
@icontract.ensure(
    lambda result, n_alphas: isinstance(result, bool) and result == (n_alphas != "deprecated"),
    "warning predicate must match the explicit n_alphas branch",
)
def cd_path_n_alphas_warning_required(n_alphas: object) -> bool:
    """Return whether lasso_path or enet_path should warn for explicit n_alphas."""
    return n_alphas != "deprecated"


@register_atom(witness_cd_path_n_alphas_warning_message)
@icontract.require(lambda function_name: _path_helper_name(function_name), "function_name must name a path helper")
@icontract.ensure(
    lambda result: result == _N_ALPHAS_DEPRECATED_MESSAGE,
    "n_alphas deprecation message must match sklearn main",
)
def cd_path_n_alphas_warning_message(function_name: str) -> str:
    """Return the shared n_alphas FutureWarning text for path helpers."""
    del function_name
    return _N_ALPHAS_DEPRECATED_MESSAGE


@register_atom(witness_cd_path_alphas_none_warning_required)
@icontract.require(lambda alphas: _valid_alphas_value(alphas), "alphas must be None, warn, or non-string")
@icontract.ensure(
    lambda result, alphas: isinstance(result, bool) and result == (alphas is None),
    "warning predicate must match the alphas=None branch",
)
def cd_path_alphas_none_warning_required(alphas: Any) -> bool:
    """Return whether lasso_path or enet_path should warn for alphas=None."""
    return alphas is None


@register_atom(witness_cd_path_alphas_none_warning_message)
@icontract.require(lambda function_name: _path_helper_name(function_name), "function_name must name a path helper")
@icontract.ensure(
    lambda result: result == _ALPHAS_NONE_DEPRECATED_MESSAGE,
    "alphas=None deprecation message must match sklearn main",
)
def cd_path_alphas_none_warning_message(function_name: str) -> str:
    """Return the shared alphas=None FutureWarning text for path helpers."""
    del function_name
    return _ALPHAS_NONE_DEPRECATED_MESSAGE


@register_atom(witness_cd_path_effective_alphas_resolution)
@icontract.require(lambda n_alphas: _n_alphas_valid(n_alphas), "n_alphas must be deprecated or positive")
@icontract.require(lambda alphas: _valid_alphas_value(alphas), "alphas must be None, warn, or non-string")
@icontract.ensure(
    lambda result: isinstance(result, dict)
    and set(result) == {"effective_alphas", "warn_n_alphas", "warn_alphas_none"},
    "result must expose the path deprecation prelude decisions",
)
def cd_path_effective_alphas_resolution(n_alphas: object, alphas: Any) -> dict[str, object]:
    """Resolve effective _alphas and warning predicates before path body logic."""
    effective_alphas: object = cd_path_default_n_alphas_resolution(n_alphas)
    warn_n_alphas = cd_path_n_alphas_warning_required(n_alphas)
    warn_alphas_none = False

    if _warn_sentinel(alphas):
        pass
    elif alphas is None:
        warn_alphas_none = True
    else:
        effective_alphas = alphas

    return {
        "effective_alphas": effective_alphas,
        "warn_n_alphas": warn_n_alphas,
        "warn_alphas_none": warn_alphas_none,
    }
