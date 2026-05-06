"""Ghost witnesses for sklearn coordinate-descent CV splitter callback-shell atoms."""

from __future__ import annotations


def witness_cd_cv_checked_cv(cv_after_check_cv: object) -> object:
    """Describe the cv object after the deferred check_cv(...) callback."""
    return cv_after_check_cv


def witness_cd_cv_split_kwargs(split_params: object) -> object:
    """Describe the kwargs expanded into cv.split(..., **routed_params.splitter.split)."""
    return split_params


def witness_cd_cv_split_iterator(split_iterator: object) -> object:
    """Describe the split iterator returned by cv.split(...) before list(...) materialization."""
    return split_iterator
