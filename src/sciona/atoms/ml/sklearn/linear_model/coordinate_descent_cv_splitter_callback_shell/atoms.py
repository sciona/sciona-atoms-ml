"""Sklearn coordinate-descent CV splitter callback-shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Iterable

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_checked_cv,
    witness_cd_cv_split_iterator,
    witness_cd_cv_split_kwargs,
)


@register_atom(witness_cd_cv_checked_cv)
@icontract.ensure(
    lambda result, cv_after_check_cv: result is cv_after_check_cv,
    "check_cv callback shell must preserve the cv object identity",
)
def cd_cv_checked_cv(cv_after_check_cv: object) -> object:
    """Return the cv object after the deferred check_cv(...) callback."""
    return cv_after_check_cv


@register_atom(witness_cd_cv_split_kwargs)
@icontract.require(lambda split_params: isinstance(split_params, dict), "split_params must be a dict")
@icontract.ensure(
    lambda result, split_params: isinstance(result, dict) and result == split_params,
    "cv.split kwargs must preserve the routed split-parameter mapping",
)
def cd_cv_split_kwargs(split_params: dict[object, object]) -> dict[object, object]:
    """Return the kwargs payload expanded into cv.split(..., **routed_params.splitter.split)."""
    return dict(split_params)


@register_atom(witness_cd_cv_split_iterator)
@icontract.require(
    lambda split_iterator: isinstance(split_iterator, Iterable),
    "split_iterator must be an iterable returned by cv.split(...)",
)
@icontract.ensure(
    lambda result, split_iterator: result is split_iterator,
    "split iterator callback shell must preserve iterator identity",
)
def cd_cv_split_iterator(split_iterator: Iterable[object]) -> Iterable[object]:
    """Return the split iterator produced by the deferred cv.split(...) callback."""
    return split_iterator
