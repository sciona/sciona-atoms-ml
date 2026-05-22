"""Ghost witnesses for sklearn LogisticRegressionCV path-result packaging atoms."""

from __future__ import annotations


def witness_logistic_cv_path_results(path_results: object) -> object:
    """Describe unzipping _log_reg_scoring_path results."""
    return path_results


def witness_logistic_cv_public_Cs(Cs: object) -> object:
    """Describe public Cs_ selection from scoring-path results."""
    return Cs


def witness_logistic_cv_coefs_paths_layout(
    coefs_paths: object,
    multi_class: str,
    n_classes: int,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> object:
    """Describe CV coefficient-path tensor layout normalization."""
    return (coefs_paths, multi_class, n_classes, n_folds, n_Cs, n_l1_ratios)


def witness_logistic_cv_n_iter_layout(
    n_iter: object,
    multi_class: str,
    n_classes: int,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> object:
    """Describe CV n_iter_ tensor layout normalization."""
    return (n_iter, multi_class, n_classes, n_folds, n_Cs, n_l1_ratios)


def witness_logistic_cv_scores_layout(scores: object, multi_class: str, n_classes: int, n_folds: int) -> object:
    """Describe CV score tensor layout normalization."""
    return (scores, multi_class, n_classes, n_folds)


def witness_logistic_cv_scores_by_class(classes: object, scores: object) -> object:
    """Describe class-keyed scores_ packaging."""
    return (classes, scores)


def witness_logistic_cv_coefs_paths_by_class(classes: object, coefs_paths: object) -> object:
    """Describe class-keyed coefs_paths_ packaging."""
    return (classes, coefs_paths)
