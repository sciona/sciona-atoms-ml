"""Ghost witnesses for sklearn LogisticRegressionCV best/refit state atoms."""

from __future__ import annotations


def witness_logistic_cv_loop_path_views(
    multi_class: str,
    cls: object,
    scores_by_class: object,
    coefs_paths_by_class: object,
    multinomial_scores: object,
    multinomial_coefs_paths: object,
) -> object:
    """Describe per-loop path views for OvR and multinomial CV selection."""
    return (multi_class, cls, scores_by_class, coefs_paths_by_class, multinomial_scores, multinomial_coefs_paths)


def witness_logistic_cv_best_flat_index(scores: object) -> object:
    """Describe flattened best-score index selection."""
    return scores


def witness_logistic_cv_best_C_l1_selection(best_index: int, Cs: object, l1_ratios: object) -> object:
    """Describe flattened best-index C and l1-ratio selection."""
    return (best_index, Cs, l1_ratios)


def witness_logistic_cv_refit_coef_init(coefs_paths: object, multi_class: str, best_index: int) -> object:
    """Describe refit coefficient initialization from best path coefficients."""
    return (coefs_paths, multi_class, best_index)


def witness_logistic_cv_nonrefit_best_indices(scores: object) -> object:
    """Describe per-fold best-index selection for non-refit CV."""
    return scores


def witness_logistic_cv_nonrefit_average_w(coefs_paths: object, best_indices: object, multi_class: str) -> object:
    """Describe non-refit coefficient averaging across fold-specific winners."""
    return (coefs_paths, best_indices, multi_class)


def witness_logistic_cv_nonrefit_average_C(best_indices: object, Cs: object) -> object:
    """Describe non-refit C averaging across fold-specific winners."""
    return (best_indices, Cs)


def witness_logistic_cv_nonrefit_average_l1_ratio(best_indices: object, Cs: object, l1_ratios: object, penalty: str) -> object:
    """Describe non-refit l1-ratio averaging for elastic-net CV."""
    return (best_indices, Cs, l1_ratios, penalty)


def witness_logistic_cv_multinomial_final_components(
    C_values: object,
    l1_ratio_values: object,
    w: object,
    n_classes: int,
    n_features: int,
    fit_intercept: bool,
) -> object:
    """Describe multinomial final component packaging from a supplied weight matrix."""
    return (C_values, l1_ratio_values, w, n_classes, n_features, fit_intercept)


def witness_logistic_cv_ovr_final_row(w: object, n_features: int, fit_intercept: bool) -> object:
    """Describe OvR final row packaging from a supplied weight vector."""
    return (w, n_features, fit_intercept)
