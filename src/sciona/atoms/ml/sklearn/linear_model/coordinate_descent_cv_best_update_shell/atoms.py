"""Sklearn coordinate-descent CV best-update-shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_best_candidate_count,
    witness_cd_cv_best_candidate_triples,
    witness_cd_cv_best_mse_improved,
    witness_cd_cv_fit_best_alpha,
    witness_cd_cv_fit_best_l1_ratio,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _finite_scalar(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _finite_or_posinf_scalar(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) or (np.isinf(numeric) and numeric > 0.0))


@register_atom(witness_cd_cv_best_candidate_triples)
@icontract.require(lambda l1_ratios: isinstance(l1_ratios, Sequence), "l1_ratios must be a sequence")
@icontract.require(lambda alphas: isinstance(alphas, Sequence), "alphas must be a sequence")
@icontract.require(lambda mean_mse: isinstance(mean_mse, Sequence), "mean_mse must be a sequence")
@icontract.require(
    lambda l1_ratios, alphas, mean_mse: len(l1_ratios) == len(alphas) == len(mean_mse) and len(l1_ratios) >= 1,
    "l1_ratios, alphas, and mean_mse must have the same positive length",
)
@icontract.ensure(
    lambda result, l1_ratios: isinstance(result, list) and len(result) == len(l1_ratios),
    "candidate triples must materialize the zipped l1_ratio paths",
)
def cd_cv_best_candidate_triples(
    l1_ratios: Sequence[object],
    alphas: Sequence[object],
    mean_mse: Sequence[object],
) -> list[tuple[object, object, object]]:
    """Return the materialized zip(l1_ratios, alphas, mean_mse) iteration shell."""
    return list(zip(l1_ratios, alphas, mean_mse))


@register_atom(witness_cd_cv_best_candidate_count)
@icontract.require(
    lambda candidate_triples: isinstance(candidate_triples, Sequence) and len(candidate_triples) >= 1,
    "candidate_triples must be a nonempty sequence",
)
@icontract.ensure(
    lambda result, candidate_triples: _positive_int(result) and int(result) == len(candidate_triples),
    "candidate count must equal len(candidate_triples)",
)
def cd_cv_best_candidate_count(candidate_triples: Sequence[object]) -> int:
    """Return the number of best-candidate triplets iterated by LinearModelCV.fit."""
    return len(candidate_triples)


@register_atom(witness_cd_cv_best_mse_improved)
@icontract.require(
    lambda best_mse: _finite_or_posinf_scalar(best_mse),
    "best_mse must be finite or positive infinity",
)
@icontract.require(lambda this_best_mse: _finite_scalar(this_best_mse), "this_best_mse must be finite")
@icontract.ensure(
    lambda result, best_mse, this_best_mse: isinstance(result, bool)
    and result == (float(this_best_mse) < float(best_mse)),
    "improvement flag must match sklearn's strict this_best_mse < best_mse guard",
)
def cd_cv_best_mse_improved(best_mse: float, this_best_mse: float) -> bool:
    """Return whether one l1-ratio path improves the running best MSE."""
    return float(this_best_mse) < float(best_mse)


@register_atom(witness_cd_cv_fit_best_l1_ratio)
@icontract.ensure(
    lambda result, best_l1_ratio: result == best_l1_ratio,
    "self.l1_ratio_ assignment must preserve the selected best_l1_ratio",
)
def cd_cv_fit_best_l1_ratio(best_l1_ratio: object) -> object:
    """Return the final l1_ratio_ value exposed by LinearModelCV.fit."""
    return best_l1_ratio


@register_atom(witness_cd_cv_fit_best_alpha)
@icontract.require(lambda best_alpha: _finite_scalar(best_alpha), "best_alpha must be finite")
@icontract.ensure(
    lambda result, best_alpha: np.isclose(float(result), float(best_alpha)),
    "self.alpha_ assignment must preserve the selected best_alpha",
)
def cd_cv_fit_best_alpha(best_alpha: float) -> float:
    """Return the final alpha_ value exposed by LinearModelCV.fit."""
    return float(best_alpha)
