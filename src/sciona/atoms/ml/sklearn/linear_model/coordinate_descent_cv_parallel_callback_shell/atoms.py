"""Sklearn coordinate-descent CV Parallel callback-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_parallel_kwargs,
    witness_cd_cv_parallel_mse_paths,
)


def _valid_verbose(value: object) -> bool:
    return isinstance(value, (bool, int))


@register_atom(witness_cd_cv_parallel_kwargs)
@icontract.require(lambda verbose: _valid_verbose(verbose), "verbose must be bool or int")
@icontract.ensure(
    lambda result, n_jobs, verbose: isinstance(result, dict)
    and result == {
        "n_jobs": n_jobs,
        "verbose": verbose,
        "prefer": "threads",
    },
    "result must match the Parallel kwargs used by LinearModelCV.fit",
)
def cd_cv_parallel_kwargs(n_jobs: object, verbose: bool | int) -> dict[str, object]:
    """Return sklearn's Parallel(...) kwarg mapping for CV path evaluation."""
    return {
        "n_jobs": n_jobs,
        "verbose": verbose,
        "prefer": "threads",
    }


@register_atom(witness_cd_cv_parallel_mse_paths)
@icontract.require(
    lambda mse_paths: isinstance(mse_paths, list),
    "mse_paths must be the materialized list returned by Parallel(...)(jobs)",
)
@icontract.ensure(
    lambda result, mse_paths: isinstance(result, list)
    and len(result) == len(mse_paths),
    "mse_paths passthrough must preserve the materialized list length",
)
def cd_cv_parallel_mse_paths(mse_paths: list[object]) -> list[object]:
    """Expose the materialized mse_paths list returned by Parallel(...)(jobs)."""
    return mse_paths
