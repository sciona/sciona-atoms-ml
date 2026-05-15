"""Ghost witnesses for sklearn LARS CV orchestration shell atoms."""

from __future__ import annotations


def witness_lars_cv_path_residues_callback_kwargs(
    precompute: object,
    method: str,
    verbose: int,
    fit_intercept: bool,
    max_iter: int,
    eps: float,
    positive: bool,
) -> dict[str, object]:
    """Describe the keyword payload passed from LarsCV.fit to _lars_path_residues."""
    return {
        "Gram": precompute,
        "copy": False,
        "method": method,
        "verbose": max(0, int(verbose) - 1),
        "fit_intercept": fit_intercept,
        "max_iter": max_iter,
        "eps": eps,
        "positive": positive,
    }
