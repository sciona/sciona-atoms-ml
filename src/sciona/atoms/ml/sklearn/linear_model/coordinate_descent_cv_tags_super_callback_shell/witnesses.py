"""Ghost witnesses for sklearn LinearModelCV tags super-callback atoms."""

from __future__ import annotations


def witness_cd_cv_tags_super_result(tags_from_super: object) -> object:
    """Describe the tags object returned by super().__sklearn_tags__()."""
    return tags_from_super


def witness_cd_cv_tags_return(tags: object) -> object:
    """Describe the final tags object returned by LinearModelCV.__sklearn_tags__."""
    return tags
