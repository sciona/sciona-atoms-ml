"""Ghost witnesses for sklearn coordinate-descent CV parallel-setup atoms."""

from __future__ import annotations


def witness_cd_cv_folds(cv_splits: object) -> object:
    """Describe the list(cv.split(...)) materialization shell in LinearModelCV.fit."""
    return cv_splits


def witness_cd_cv_fold_count(folds: object) -> object:
    """Describe the len(folds) bookkeeping in LinearModelCV.fit."""
    return folds


def witness_cd_cv_path_job_kwargs(this_alphas: object, this_l1_ratio: object, x_dtype_type: object) -> object:
    """Describe one delayed(_path_residuals) kwarg bundle in LinearModelCV.fit."""
    return this_alphas, this_l1_ratio, x_dtype_type


def witness_cd_cv_path_job_count(l1_ratios: object, folds: object) -> object:
    """Describe the nested l1-ratio x fold job-count shell in LinearModelCV.fit."""
    return l1_ratios, folds


def witness_cd_cv_best_mse_initial(fold_count: object) -> object:
    """Describe the best_mse = np.inf initialization in LinearModelCV.fit."""
    return fold_count
