"""Ghost witnesses for sklearn LogisticRegressionCV l1-axis packaging atoms."""

from __future__ import annotations


def witness_logistic_cv_l1_axis_enabled(public_l1_ratios_param: object) -> bool:
    """Describe the public l1-ratio branch predicate."""
    return public_l1_ratios_param is not None


def witness_logistic_cv_coefs_paths_l1_axis(coefs_paths: object, n_folds: int, n_Cs: int, n_l1_ratios: int) -> object:
    """Describe per-class coefficient-path public l1-axis reshaping."""
    return (coefs_paths, n_folds, n_Cs, n_l1_ratios)


def witness_logistic_cv_coefs_paths_dict_l1_axis(
    coefs_paths_by_class: object,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> object:
    """Describe class-keyed coefficient-path public l1-axis reshaping."""
    return (coefs_paths_by_class, n_folds, n_Cs, n_l1_ratios)


def witness_logistic_cv_scores_l1_axis(scores: object, n_folds: int, n_Cs: int, n_l1_ratios: int) -> object:
    """Describe per-class score public l1-axis reshaping."""
    return (scores, n_folds, n_Cs, n_l1_ratios)


def witness_logistic_cv_scores_dict_l1_axis(scores_by_class: object, n_folds: int, n_Cs: int, n_l1_ratios: int) -> object:
    """Describe class-keyed score public l1-axis reshaping."""
    return (scores_by_class, n_folds, n_Cs, n_l1_ratios)


def witness_logistic_cv_n_iter_l1_axis(n_iter: object, n_folds: int, n_Cs: int, n_l1_ratios: int) -> object:
    """Describe n_iter_ public l1-axis reshaping."""
    return (n_iter, n_folds, n_Cs, n_l1_ratios)
