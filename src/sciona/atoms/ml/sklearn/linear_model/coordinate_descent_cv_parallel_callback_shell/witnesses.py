"""Ghost witnesses for sklearn coordinate-descent CV Parallel callback-shell atoms."""

from __future__ import annotations


def witness_cd_cv_parallel_kwargs(n_jobs: object, verbose: object) -> object:
    """Describe the Parallel(...) kwarg mapping in LinearModelCV.fit."""
    return n_jobs, verbose


def witness_cd_cv_parallel_mse_paths(mse_paths: object) -> object:
    """Describe the mse_paths list returned by Parallel(...)(jobs) in LinearModelCV.fit."""
    return mse_paths
