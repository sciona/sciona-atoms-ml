"""Ghost witnesses for sklearn coordinate-descent CV subclass API-shell atoms."""

from __future__ import annotations


def witness_cd_cv_subclass_path_name(cv_kind: object) -> object:
    """Describe the static path helper selected by a CV subclass."""
    return cv_kind


def witness_cd_cv_subclass_estimator_name(cv_kind: object) -> object:
    """Describe the concrete estimator selected by a CV subclass."""
    return cv_kind


def witness_cd_cv_subclass_is_multitask(cv_kind: object) -> object:
    """Describe the multitask flag returned by a CV subclass."""
    return cv_kind


def witness_cd_cv_subclass_target_single_output_tag(multitask: object) -> object:
    """Describe the multitask subclass single-output target tag override."""
    return multitask


def witness_cd_cv_subclass_fit_forwards_sample_weight(multitask: object) -> object:
    """Describe whether subclass fit forwards sample_weight to LinearModelCV.fit."""
    return multitask


def witness_cd_cv_subclass_super_fit_args(X: object, y: object) -> object:
    """Describe positional args passed into super().fit(X, y, ...)."""
    return X, y


def witness_cd_cv_subclass_super_fit_kwargs(
    params: object, sample_weight: object, forwards_sample_weight: object
) -> object:
    """Describe keyword args passed into subclass super().fit(...)."""
    return params, sample_weight, forwards_sample_weight
