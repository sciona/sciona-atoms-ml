"""Ghost witnesses for sklearn coordinate-descent CV best-update-shell atoms."""

from __future__ import annotations


def witness_cd_cv_best_candidate_triples(l1_ratios: object, alphas: object, mean_mse: object) -> object:
    """Describe the zip(l1_ratios, alphas, mean_mse) iteration shell in LinearModelCV.fit."""
    return l1_ratios, alphas, mean_mse


def witness_cd_cv_best_candidate_count(candidate_triples: object) -> object:
    """Describe the number of best-candidate triplets iterated in LinearModelCV.fit."""
    return candidate_triples


def witness_cd_cv_best_mse_improved(best_mse: object, this_best_mse: object) -> object:
    """Describe the this_best_mse < best_mse guard in LinearModelCV.fit."""
    return best_mse, this_best_mse


def witness_cd_cv_fit_best_l1_ratio(best_l1_ratio: object) -> object:
    """Describe the final self.l1_ratio_ assignment shell in LinearModelCV.fit."""
    return best_l1_ratio


def witness_cd_cv_fit_best_alpha(best_alpha: object) -> object:
    """Describe the final self.alpha_ assignment shell in LinearModelCV.fit."""
    return best_alpha
