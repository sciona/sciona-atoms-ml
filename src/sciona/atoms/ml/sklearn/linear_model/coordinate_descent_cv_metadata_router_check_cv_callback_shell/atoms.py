"""Sklearn LinearModelCV metadata-router check_cv callback atoms."""

from __future__ import annotations

from collections.abc import Iterable

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_metadata_router_check_cv_args,
    witness_cd_cv_metadata_router_checked_splitter_result,
)


def _check_cv_payload(value: object) -> bool:
    return (
        value is None
        or isinstance(value, int)
        or (hasattr(value, "split") and not isinstance(value, (str, bytes, bytearray)))
        or (isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)))
    )


@register_atom(witness_cd_cv_metadata_router_check_cv_args)
@icontract.require(lambda cv: _check_cv_payload(cv), "cv must be a check_cv payload")
@icontract.ensure(
    lambda result, cv: isinstance(result, tuple) and len(result) == 1 and result[0] is cv,
    "get_metadata_routing check_cv args must preserve self.cv",
)
def cd_cv_metadata_router_check_cv_args(cv: object) -> tuple[object]:
    """Return positional args for check_cv(self.cv) in get_metadata_routing."""
    return (cv,)


@register_atom(witness_cd_cv_metadata_router_checked_splitter_result)
@icontract.require(lambda checked_splitter: checked_splitter is not None, "checked_splitter must be provided")
@icontract.ensure(
    lambda result, checked_splitter: result is checked_splitter,
    "checked splitter result must preserve callback output identity",
)
def cd_cv_metadata_router_checked_splitter_result(checked_splitter: object) -> object:
    """Return the checked splitter produced by check_cv(self.cv)."""
    return checked_splitter
