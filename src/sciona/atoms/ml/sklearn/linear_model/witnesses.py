"""Ghost witnesses for sklearn linear model atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import (
    ARDRegressionState,
    BayesianRidgeState,
    LarsPathState,
    LarsState,
    LinearRegressionState,
    OrthogonalMatchingPursuitCVState,
    OrthogonalMatchingPursuitState,
    RidgeClassifierCVState,
    RidgeClassifierState,
    RidgeCVState,
    RidgeState,
    TheilSenRegressorState,
)


def witness_linear_regression_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    fit_intercept: bool = True,
    copy_X: bool = True,
    tol: float = 1e-6,
    n_jobs: int | None = None,
    positive: bool = False,
    sample_weight: float | tuple[float, ...] | None = None,
) -> AbstractArray:
    """Describe fitting dense ordinary least-squares coefficients."""
    del copy_X, tol, n_jobs, sample_weight
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept must be boolean")
    if positive:
        raise ValueError("positive=True is outside the dense OLS atom scope")
    n_outputs = 1 if len(y.shape) == 1 else int(y.shape[1])
    if n_outputs == 1:
        return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")
    return AbstractArray(shape=(n_outputs, int(X.shape[1])), dtype="float64")


def witness_linear_regression_predict(X: AbstractArray, state: LinearRegressionState) -> AbstractArray:
    """Describe predicting with fitted ordinary least-squares coefficients."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if state.n_outputs == 1:
        return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")
    return AbstractArray(shape=(int(X.shape[0]), state.n_outputs), dtype="float64")


def witness_ridge_regression(
    X: AbstractArray,
    y: AbstractArray,
    alpha: float | tuple[float, ...] = 1.0,
    *,
    sample_weight: float | tuple[float, ...] | None = None,
    solver: str = "auto",
    max_iter: int | None = None,
    tol: float = 1e-4,
    positive: bool = False,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe solving dense ridge-regression coefficients."""
    del sample_weight, max_iter, tol, random_state
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if solver not in {"auto", "cholesky"}:
        raise ValueError("only dense cholesky ridge solving is covered")
    if positive:
        raise ValueError("positive=True is outside this dense ridge atom scope")
    n_outputs = 1 if len(y.shape) == 1 else int(y.shape[1])
    alpha_values = alpha if isinstance(alpha, tuple) else (alpha,)
    if len(alpha_values) not in {1, n_outputs}:
        raise ValueError("alpha must be scalar or match output count")
    if n_outputs == 1:
        return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")
    return AbstractArray(shape=(n_outputs, int(X.shape[1])), dtype="float64")


def witness_ridge_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    alpha: float | tuple[float, ...] = 1.0,
    fit_intercept: bool = True,
    copy_X: bool = True,
    max_iter: int | None = None,
    tol: float = 1e-4,
    solver: str = "auto",
    positive: bool = False,
    random_state: int | None = None,
    sample_weight: float | tuple[float, ...] | None = None,
) -> AbstractArray:
    """Describe fitting dense ridge-regression coefficients."""
    del copy_X, max_iter, tol, random_state, sample_weight
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept must be boolean")
    if solver not in {"auto", "cholesky"}:
        raise ValueError("only dense cholesky ridge fitting is covered")
    if positive:
        raise ValueError("positive=True is outside this dense ridge atom scope")
    n_outputs = 1 if len(y.shape) == 1 else int(y.shape[1])
    alpha_values = alpha if isinstance(alpha, tuple) else (alpha,)
    if len(alpha_values) not in {1, n_outputs}:
        raise ValueError("alpha must be scalar or match output count")
    if n_outputs == 1:
        return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")
    return AbstractArray(shape=(n_outputs, int(X.shape[1])), dtype="float64")


def witness_ridge_predict(X: AbstractArray, state: RidgeState) -> AbstractArray:
    """Describe predicting with fitted dense ridge-regression coefficients."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if state.n_outputs == 1:
        return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")
    return AbstractArray(shape=(int(X.shape[0]), state.n_outputs), dtype="float64")


def witness_ridge_cv_scores(
    X: AbstractArray,
    y: AbstractArray,
    alphas: float | tuple[float, ...] = (0.1, 1.0, 10.0),
    *,
    fit_intercept: bool = True,
    scoring: None = None,
    cv: None = None,
    sample_weight: None = None,
) -> AbstractArray:
    """Describe dense leave-one-out ridge CV scores across alphas."""
    del fit_intercept, scoring, cv, sample_weight
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if X.shape[0] < 2:
        raise ValueError("leave-one-out CV requires at least two samples")
    alpha_values = alphas if isinstance(alphas, tuple) else (alphas,)
    return AbstractArray(shape=(len(alpha_values),), dtype="float64")


def witness_ridge_cv_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    alphas: float | tuple[float, ...] = (0.1, 1.0, 10.0),
    fit_intercept: bool = True,
    scoring: None = None,
    cv: None = None,
    gcv_mode: str | None = None,
    store_cv_results: bool = False,
    alpha_per_target: bool = False,
    sample_weight: None = None,
) -> AbstractArray:
    """Describe fitting dense ridge regression after LOO alpha selection."""
    del alphas, scoring, cv, gcv_mode, store_cv_results, alpha_per_target, sample_weight
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if X.shape[0] < 2:
        raise ValueError("leave-one-out CV requires at least two samples")
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept must be boolean")
    n_outputs = 1 if len(y.shape) == 1 else int(y.shape[1])
    if n_outputs == 1:
        return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")
    return AbstractArray(shape=(n_outputs, int(X.shape[1])), dtype="float64")


def witness_ridge_cv_predict(X: AbstractArray, state: RidgeCVState) -> AbstractArray:
    """Describe predicting with fitted dense RidgeCV coefficients."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if state.n_outputs == 1:
        return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")
    return AbstractArray(shape=(int(X.shape[0]), state.n_outputs), dtype="float64")


def witness_ridge_classifier_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    alpha: float | tuple[float, ...] = 1.0,
    fit_intercept: bool = True,
    copy_X: bool = True,
    max_iter: int | None = None,
    tol: float = 1e-4,
    class_weight: dict[float, float] | str | None = None,
    solver: str = "auto",
    positive: bool = False,
    random_state: int | None = None,
    sample_weight: float | tuple[float, ...] | None = None,
) -> AbstractArray:
    """Describe fitting dense ridge-classifier coefficients."""
    del alpha, copy_X, max_iter, tol, class_weight, random_state, sample_weight
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept must be boolean")
    if solver not in {"auto", "cholesky"}:
        raise ValueError("only dense cholesky ridge classification is covered")
    if positive:
        raise ValueError("positive=True is outside this dense ridge classifier atom scope")
    return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")


def witness_ridge_classifier_decision_function(X: AbstractArray, state: RidgeClassifierState) -> AbstractArray:
    """Describe dense ridge-classifier confidence scores."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if state.classes.shape[0] == 2:
        return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")
    return AbstractArray(shape=(int(X.shape[0]), int(state.classes.shape[0])), dtype="float64")


def witness_ridge_classifier_predict(X: AbstractArray, state: RidgeClassifierState) -> AbstractArray:
    """Describe dense ridge-classifier label prediction."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")


def witness_ridge_classifier_cv_scores(
    X: AbstractArray,
    y: AbstractArray,
    alphas: float | tuple[float, ...] = (0.1, 1.0, 10.0),
    *,
    fit_intercept: bool = True,
    scoring: None = None,
    cv: None = None,
    class_weight: None = None,
    sample_weight: None = None,
) -> AbstractArray:
    """Describe dense leave-one-out RidgeClassifierCV scores across alphas."""
    del fit_intercept, scoring, cv, class_weight, sample_weight
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if X.shape[0] < 2:
        raise ValueError("leave-one-out CV requires at least two samples")
    alpha_values = alphas if isinstance(alphas, tuple) else (alphas,)
    return AbstractArray(shape=(len(alpha_values),), dtype="float64")


def witness_ridge_classifier_cv_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    alphas: float | tuple[float, ...] = (0.1, 1.0, 10.0),
    fit_intercept: bool = True,
    scoring: None = None,
    cv: None = None,
    class_weight: None = None,
    store_cv_results: bool = False,
    sample_weight: None = None,
) -> AbstractArray:
    """Describe fitting dense ridge classification after LOO alpha selection."""
    del alphas, scoring, cv, class_weight, store_cv_results, sample_weight
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if X.shape[0] < 2:
        raise ValueError("leave-one-out CV requires at least two samples")
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept must be boolean")
    return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")


def witness_ridge_classifier_cv_decision_function(X: AbstractArray, state: RidgeClassifierCVState) -> AbstractArray:
    """Describe dense RidgeClassifierCV confidence scores."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if state.classes.shape[0] == 2:
        return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")
    return AbstractArray(shape=(int(X.shape[0]), int(state.classes.shape[0])), dtype="float64")


def witness_ridge_classifier_cv_predict(X: AbstractArray, state: RidgeClassifierCVState) -> AbstractArray:
    """Describe dense RidgeClassifierCV label prediction."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")


def witness_orthogonal_mp(
    X: AbstractArray,
    y: AbstractArray,
    *,
    n_nonzero_coefs: int | None = None,
    tol: float | None = None,
    precompute: bool | str = False,
    copy_X: bool = True,
    return_path: bool = False,
    return_n_iter: bool = False,
) -> AbstractArray:
    """Describe dense OMP coefficient solving from a design matrix."""
    del n_nonzero_coefs, tol, precompute, copy_X
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if return_path or return_n_iter:
        raise ValueError("path and iteration-return modes are outside this atom scope")
    if len(y.shape) == 1:
        return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")
    return AbstractArray(shape=(int(X.shape[1]), int(y.shape[1])), dtype="float64")


def witness_orthogonal_mp_gram(
    Gram: AbstractArray,
    Xy: AbstractArray,
    *,
    n_nonzero_coefs: int | None = None,
    tol: float | None = None,
    norms_squared: tuple[float, ...] | None = None,
    copy_Gram: bool = True,
    copy_Xy: bool = True,
    return_path: bool = False,
    return_n_iter: bool = False,
) -> AbstractArray:
    """Describe dense OMP coefficient solving from Gram inputs."""
    del n_nonzero_coefs, tol, norms_squared, copy_Gram, copy_Xy
    if len(Gram.shape) != 2 or Gram.shape[0] != Gram.shape[1]:
        raise ValueError("Gram must be square")
    if len(Xy.shape) not in {1, 2}:
        raise ValueError("Xy must be 1D or 2D")
    if Xy.shape[0] != Gram.shape[0]:
        raise ValueError("Xy feature count must match Gram")
    if return_path or return_n_iter:
        raise ValueError("path and iteration-return modes are outside this atom scope")
    if len(Xy.shape) == 1:
        return AbstractArray(shape=(int(Gram.shape[0]),), dtype="float64")
    return AbstractArray(shape=(int(Gram.shape[0]), int(Xy.shape[1])), dtype="float64")


def witness_orthogonal_matching_pursuit_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    n_nonzero_coefs: int | None = None,
    tol: float | None = None,
    fit_intercept: bool = True,
    precompute: bool | str = "auto",
) -> AbstractArray:
    """Describe fitting dense orthogonal matching pursuit coefficients."""
    del n_nonzero_coefs, tol, precompute
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept must be boolean")
    n_outputs = 1 if len(y.shape) == 1 else int(y.shape[1])
    if n_outputs == 1:
        return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")
    return AbstractArray(shape=(n_outputs, int(X.shape[1])), dtype="float64")


def witness_orthogonal_matching_pursuit_predict(X: AbstractArray, state: OrthogonalMatchingPursuitState) -> AbstractArray:
    """Describe predicting with fitted orthogonal matching pursuit coefficients."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if state.n_outputs == 1:
        return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")
    return AbstractArray(shape=(int(X.shape[0]), state.n_outputs), dtype="float64")


def witness_omp_path_residues(
    X_train: AbstractArray,
    y_train: AbstractArray,
    X_test: AbstractArray,
    y_test: AbstractArray,
    *,
    copy: bool = True,
    fit_intercept: bool = True,
    max_iter: int = 100,
) -> AbstractArray:
    """Describe OMP coefficient-path residuals on held-out data."""
    del copy, fit_intercept
    if len(X_train.shape) != 2 or len(X_test.shape) != 2:
        raise ValueError("train and test X must be 2D")
    if len(y_train.shape) != 1 or len(y_test.shape) != 1:
        raise ValueError("train and test y must be 1D")
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        raise ValueError("X and y sample counts must match")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("train and test feature counts must match")
    return AbstractArray(shape=(max_iter, int(X_test.shape[0])), dtype="float64")


def witness_orthogonal_matching_pursuit_cv_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    copy: bool = True,
    fit_intercept: bool = True,
    max_iter: int | None = None,
    cv: int | None = None,
    n_jobs: None = None,
    verbose: bool = False,
) -> AbstractArray:
    """Describe fitting dense OMP with cross-validated sparsity."""
    del copy, max_iter, cv, n_jobs, verbose
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept must be boolean")
    return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")


def witness_orthogonal_matching_pursuit_cv_predict(X: AbstractArray, state: OrthogonalMatchingPursuitCVState) -> AbstractArray:
    """Describe predicting with fitted cross-validated OMP coefficients."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")


def witness_bayesian_ridge_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    max_iter: int = 300,
    tol: float = 1e-3,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
    alpha_init: float | None = None,
    lambda_init: float | None = None,
    compute_score: bool = False,
    fit_intercept: bool = True,
    copy_X: bool = True,
    sample_weight: float | tuple[float, ...] | None = None,
) -> AbstractArray:
    """Describe fitting dense Bayesian ridge posterior state."""
    del tol, alpha_1, alpha_2, lambda_1, lambda_2, alpha_init, lambda_init, copy_X, sample_weight
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if not isinstance(compute_score, bool) or not isinstance(fit_intercept, bool):
        raise ValueError("boolean options must be boolean")
    return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")


def witness_bayesian_ridge_predict(
    X: AbstractArray,
    state: BayesianRidgeState,
    *,
    return_std: bool = False,
) -> AbstractArray:
    """Describe Bayesian ridge posterior mean predictions."""
    del return_std
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")


def witness_bayesian_ridge_predict_std(X: AbstractArray, state: BayesianRidgeState) -> AbstractArray:
    """Describe Bayesian ridge posterior predictive standard deviations."""
    return witness_bayesian_ridge_predict(X, state)


def witness_ard_regression_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    max_iter: int = 300,
    tol: float = 1e-3,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
    compute_score: bool = False,
    threshold_lambda: float = 1e4,
    fit_intercept: bool = True,
    copy_X: bool = True,
    verbose: bool = False,
) -> AbstractArray:
    """Describe fitting dense automatic relevance determination state."""
    del tol, alpha_1, alpha_2, lambda_1, lambda_2, copy_X, verbose
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if threshold_lambda <= 0:
        raise ValueError("threshold_lambda must be positive")
    if not isinstance(compute_score, bool) or not isinstance(fit_intercept, bool):
        raise ValueError("boolean options must be boolean")
    return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")


def witness_ard_regression_predict(
    X: AbstractArray,
    state: ARDRegressionState,
    *,
    return_std: bool = False,
) -> AbstractArray:
    """Describe ARD posterior mean predictions."""
    del return_std
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")


def witness_ard_regression_predict_std(X: AbstractArray, state: ARDRegressionState) -> AbstractArray:
    """Describe ARD posterior predictive standard deviations."""
    return witness_ard_regression_predict(X, state)


def witness_theil_sen_regressor_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    fit_intercept: bool = True,
    max_subpopulation: int = 10000,
    n_subsamples: int | None = None,
    max_iter: int = 300,
    tol: float = 1e-3,
    random_state: int | None = None,
    n_jobs: int | None = None,
    verbose: bool = False,
) -> AbstractArray:
    """Describe fitting dense Theil-Sen regression state."""
    del tol, random_state, n_jobs, verbose
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept must be boolean")
    if max_subpopulation < 1:
        raise ValueError("max_subpopulation must be positive")
    if n_subsamples is not None and n_subsamples < 1:
        raise ValueError("n_subsamples must be positive when provided")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")


def witness_theil_sen_regressor_predict(X: AbstractArray, state: TheilSenRegressorState) -> AbstractArray:
    """Describe predicting with fitted dense Theil-Sen coefficients."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")


def witness_lars_path(
    X: AbstractArray,
    y: AbstractArray,
    Xy: AbstractArray | None = None,
    *,
    Gram: AbstractArray | str | bool | None = None,
    max_iter: int = 500,
    alpha_min: float = 0.0,
    method: str = "lar",
    copy_X: bool = True,
    eps: float = 2.220446049250313e-16,
    copy_Gram: bool = True,
    verbose: int | bool = 0,
    return_path: bool = True,
    return_n_iter: bool = False,
    positive: bool = False,
) -> AbstractArray:
    """Describe dense least-angle-regression path coefficients."""
    del Xy, Gram, copy_X, eps, copy_Gram, verbose, return_n_iter
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if alpha_min < 0.0:
        raise ValueError("alpha_min must be non-negative")
    if method != "lar" or positive:
        raise ValueError("only unconstrained method='lar' is covered")
    if not return_path:
        raise ValueError("return_path=False is outside this atom scope")
    return AbstractArray(shape=(int(X.shape[1]), min(int(X.shape[1]), max_iter) + 1), dtype="float64")


def witness_lars_path_gram(
    Xy: AbstractArray,
    Gram: AbstractArray,
    *,
    n_samples: int,
    max_iter: int = 500,
    alpha_min: float = 0.0,
    method: str = "lar",
    copy_X: bool = True,
    eps: float = 2.220446049250313e-16,
    copy_Gram: bool = True,
    verbose: int | bool = 0,
    return_path: bool = True,
    return_n_iter: bool = False,
    positive: bool = False,
) -> AbstractArray:
    """Describe sufficient-statistics least-angle-regression path coefficients."""
    del copy_X, eps, copy_Gram, verbose, return_n_iter
    if len(Gram.shape) != 2 or Gram.shape[0] != Gram.shape[1]:
        raise ValueError("Gram must be square")
    if len(Xy.shape) != 1 or Xy.shape[0] != Gram.shape[0]:
        raise ValueError("Xy must be 1D and match Gram")
    if n_samples < 1 or max_iter < 1:
        raise ValueError("sample count and max_iter must be positive")
    if alpha_min < 0.0:
        raise ValueError("alpha_min must be non-negative")
    if method != "lar" or positive:
        raise ValueError("only unconstrained method='lar' is covered")
    if not return_path:
        raise ValueError("return_path=False is outside this atom scope")
    return AbstractArray(shape=(int(Gram.shape[0]), min(int(Gram.shape[0]), max_iter) + 1), dtype="float64")


def witness_lars_fit(
    X: AbstractArray,
    y: AbstractArray,
    Xy: AbstractArray | None = None,
    *,
    fit_intercept: bool = True,
    verbose: int | bool = False,
    precompute: bool | str = "auto",
    n_nonzero_coefs: int = 500,
    eps: float = 2.220446049250313e-16,
    copy_X: bool = True,
    fit_path: bool = True,
    jitter: float | None = None,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe fitting dense single-output LARS coefficients."""
    del Xy, verbose, precompute, eps, copy_X, random_state
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts")
    if not isinstance(fit_intercept, bool) or not isinstance(fit_path, bool):
        raise ValueError("boolean options must be boolean")
    if n_nonzero_coefs < 1:
        raise ValueError("n_nonzero_coefs must be positive")
    if jitter is not None:
        raise ValueError("jitter is outside this atom scope")
    return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")


def witness_lars_predict(X: AbstractArray, state: LarsState) -> AbstractArray:
    """Describe predicting with fitted dense LARS coefficients."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="float64")
