"""Ghost witnesses for sklearn LinearModelCV metadata-router check_cv callback atoms."""

from __future__ import annotations


def witness_cd_cv_metadata_router_check_cv_args(cv: object) -> object:
    """Describe the check_cv(self.cv) positional payload in get_metadata_routing."""
    return cv


def witness_cd_cv_metadata_router_checked_splitter_result(checked_splitter: object) -> object:
    """Describe the checked CV splitter callback output in get_metadata_routing."""
    return checked_splitter
