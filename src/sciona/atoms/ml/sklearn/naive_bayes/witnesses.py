"""Ghost witnesses for sklearn naive Bayes atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import BernoulliNBState, CategoricalNBState, ComplementNBState, GaussianNBState, MultinomialNBState


def witness_gaussian_nb_update_mean_variance(
    n_past: float,
    mu: AbstractArray,
    var: AbstractArray,
    X: AbstractArray,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe one online Gaussian mean and variance update."""
    if n_past < 0:
        raise ValueError("n_past must be nonnegative")
    if len(mu.shape) != 1 or len(var.shape) != 1:
        raise ValueError("mu and var must be 1D")
    if mu.shape != var.shape:
        raise ValueError("mu and var must have matching shape")
    if len(X.shape) != 2 or X.shape[1] != mu.shape[0]:
        raise ValueError("X must be 2D with the same feature count as mu")
    if sample_weight is not None and sample_weight.shape != (X.shape[0],):
        raise ValueError("sample_weight must match the row count of X")
    return AbstractArray(shape=(2, int(mu.shape[0])), dtype="float64")


def witness_gaussian_nb_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    priors: AbstractArray | None = None,
    var_smoothing: float = 1e-9,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe dense Gaussian naive Bayes state learned from labels."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if y.shape != (X.shape[0],):
        raise ValueError("y must match the row count of X")
    if priors is not None and len(priors.shape) != 1:
        raise ValueError("priors must be 1D")
    if sample_weight is not None and sample_weight.shape != (X.shape[0],):
        raise ValueError("sample_weight must match the row count of X")
    if var_smoothing < 0:
        raise ValueError("var_smoothing must be nonnegative")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_gaussian_nb_joint_log_likelihood(X: AbstractArray, state: GaussianNBState) -> AbstractArray:
    """Describe Gaussian class joint log likelihoods for each row."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]), int(state.classes.shape[0])), dtype="float64")


def witness_gaussian_nb_predict_log_proba(X: AbstractArray, state: GaussianNBState) -> AbstractArray:
    """Describe normalized Gaussian log probabilities for each class."""
    return witness_gaussian_nb_joint_log_likelihood(X, state)


def witness_gaussian_nb_predict_proba(X: AbstractArray, state: GaussianNBState) -> AbstractArray:
    """Describe normalized Gaussian probabilities for each class."""
    return witness_gaussian_nb_joint_log_likelihood(X, state)


def witness_gaussian_nb_predict(X: AbstractArray, state: GaussianNBState) -> AbstractArray:
    """Describe one integer class prediction per input row."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="int64")


def witness_multinomial_nb_count(
    X: AbstractArray,
    y: AbstractArray,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe class-feature count accumulation for multinomial features."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if y.shape != (X.shape[0],):
        raise ValueError("y must match the row count of X")
    if sample_weight is not None and sample_weight.shape != (X.shape[0],):
        raise ValueError("sample_weight must match the row count of X")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_multinomial_nb_feature_log_prob(feature_count: AbstractArray, *, alpha: float = 1.0) -> AbstractArray:
    """Describe smoothed multinomial feature log probabilities."""
    if len(feature_count.shape) != 2:
        raise ValueError("feature_count must be 2D")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return AbstractArray(shape=feature_count.shape, dtype="float64")


def witness_multinomial_nb_class_log_prior(
    class_count: AbstractArray,
    *,
    fit_prior: bool = True,
    class_prior: AbstractArray | None = None,
) -> AbstractArray:
    """Describe class log priors for discrete naive Bayes."""
    if len(class_count.shape) != 1:
        raise ValueError("class_count must be 1D")
    if class_prior is not None and class_prior.shape != class_count.shape:
        raise ValueError("class_prior must match class_count")
    return AbstractArray(shape=class_count.shape, dtype="float64")


def witness_multinomial_nb_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    alpha: float = 1.0,
    fit_prior: bool = True,
    class_prior: AbstractArray | None = None,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe dense multinomial naive Bayes state learned from labels."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if y.shape != (X.shape[0],):
        raise ValueError("y must match the row count of X")
    if class_prior is not None and len(class_prior.shape) != 1:
        raise ValueError("class_prior must be 1D")
    if sample_weight is not None and sample_weight.shape != (X.shape[0],):
        raise ValueError("sample_weight must match the row count of X")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_multinomial_nb_joint_log_likelihood(X: AbstractArray, state: MultinomialNBState) -> AbstractArray:
    """Describe multinomial class joint log likelihoods for each row."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]), int(state.classes.shape[0])), dtype="float64")


def witness_multinomial_nb_predict_log_proba(X: AbstractArray, state: MultinomialNBState) -> AbstractArray:
    """Describe normalized multinomial log probabilities for each class."""
    return witness_multinomial_nb_joint_log_likelihood(X, state)


def witness_multinomial_nb_predict_proba(X: AbstractArray, state: MultinomialNBState) -> AbstractArray:
    """Describe normalized multinomial probabilities for each class."""
    return witness_multinomial_nb_joint_log_likelihood(X, state)


def witness_multinomial_nb_predict(X: AbstractArray, state: MultinomialNBState) -> AbstractArray:
    """Describe one integer class prediction per multinomial input row."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="int64")


def witness_complement_nb_count(
    X: AbstractArray,
    y: AbstractArray,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe complement naive Bayes class-feature count accumulation."""
    return witness_multinomial_nb_count(X, y, sample_weight)


def witness_complement_nb_feature_log_prob(
    feature_count: AbstractArray,
    feature_all: AbstractArray,
    *,
    alpha: float = 1.0,
    norm: bool = False,
) -> AbstractArray:
    """Describe complement-smoothed feature weights."""
    if len(feature_count.shape) != 2:
        raise ValueError("feature_count must be 2D")
    if feature_all.shape != (feature_count.shape[1],):
        raise ValueError("feature_all must match the feature count")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return AbstractArray(shape=feature_count.shape, dtype="float64")


def witness_complement_nb_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    alpha: float = 1.0,
    fit_prior: bool = True,
    class_prior: AbstractArray | None = None,
    norm: bool = False,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe dense complement naive Bayes state learned from labels."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if y.shape != (X.shape[0],):
        raise ValueError("y must match the row count of X")
    if class_prior is not None and len(class_prior.shape) != 1:
        raise ValueError("class_prior must be 1D")
    if sample_weight is not None and sample_weight.shape != (X.shape[0],):
        raise ValueError("sample_weight must match the row count of X")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_complement_nb_joint_log_likelihood(X: AbstractArray, state: ComplementNBState) -> AbstractArray:
    """Describe complement class scores for each row."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]), int(state.classes.shape[0])), dtype="float64")


def witness_complement_nb_predict_log_proba(X: AbstractArray, state: ComplementNBState) -> AbstractArray:
    """Describe normalized complement log scores for each class."""
    return witness_complement_nb_joint_log_likelihood(X, state)


def witness_complement_nb_predict_proba(X: AbstractArray, state: ComplementNBState) -> AbstractArray:
    """Describe normalized complement probabilities for each class."""
    return witness_complement_nb_joint_log_likelihood(X, state)


def witness_complement_nb_predict(X: AbstractArray, state: ComplementNBState) -> AbstractArray:
    """Describe one integer class prediction per complement input row."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="int64")


def witness_bernoulli_nb_binarize(X: AbstractArray, *, binarize: float | None = 0.0) -> AbstractArray:
    """Describe Bernoulli thresholding of dense input features."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_bernoulli_nb_count(
    X: AbstractArray,
    y: AbstractArray,
    *,
    binarize: float | None = 0.0,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe Bernoulli class-feature count accumulation."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if y.shape != (X.shape[0],):
        raise ValueError("y must match the row count of X")
    if sample_weight is not None and sample_weight.shape != (X.shape[0],):
        raise ValueError("sample_weight must match the row count of X")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_bernoulli_nb_feature_log_prob(
    feature_count: AbstractArray,
    class_count: AbstractArray,
    *,
    alpha: float = 1.0,
) -> AbstractArray:
    """Describe smoothed Bernoulli feature log probabilities."""
    if len(feature_count.shape) != 2 or len(class_count.shape) != 1:
        raise ValueError("feature_count must be 2D and class_count must be 1D")
    if feature_count.shape[0] != class_count.shape[0]:
        raise ValueError("class dimensions must match")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return AbstractArray(shape=feature_count.shape, dtype="float64")


def witness_bernoulli_nb_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    alpha: float = 1.0,
    binarize: float | None = 0.0,
    fit_prior: bool = True,
    class_prior: AbstractArray | None = None,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe dense Bernoulli naive Bayes state learned from labels."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if y.shape != (X.shape[0],):
        raise ValueError("y must match the row count of X")
    if class_prior is not None and len(class_prior.shape) != 1:
        raise ValueError("class_prior must be 1D")
    if sample_weight is not None and sample_weight.shape != (X.shape[0],):
        raise ValueError("sample_weight must match the row count of X")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_bernoulli_nb_joint_log_likelihood(X: AbstractArray, state: BernoulliNBState) -> AbstractArray:
    """Describe Bernoulli class joint log likelihoods for each row."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]), int(state.classes.shape[0])), dtype="float64")


def witness_bernoulli_nb_predict_log_proba(X: AbstractArray, state: BernoulliNBState) -> AbstractArray:
    """Describe normalized Bernoulli log probabilities for each class."""
    return witness_bernoulli_nb_joint_log_likelihood(X, state)


def witness_bernoulli_nb_predict_proba(X: AbstractArray, state: BernoulliNBState) -> AbstractArray:
    """Describe normalized Bernoulli probabilities for each class."""
    return witness_bernoulli_nb_joint_log_likelihood(X, state)


def witness_bernoulli_nb_predict(X: AbstractArray, state: BernoulliNBState) -> AbstractArray:
    """Describe one integer class prediction per Bernoulli input row."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="int64")


def witness_categorical_nb_n_categories(
    X: AbstractArray,
    min_categories: AbstractArray | int | None = None,
) -> AbstractArray:
    """Describe one category cardinality per categorical feature."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if min_categories is not None and not isinstance(min_categories, int) and min_categories.shape != (X.shape[1],):
        raise ValueError("min_categories must match feature count")
    return AbstractArray(shape=(int(X.shape[1]),), dtype="int64")


def witness_categorical_nb_count(
    X: AbstractArray,
    y: AbstractArray,
    min_categories: AbstractArray | int | None = None,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe categorical class and per-feature category counts."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if y.shape != (X.shape[0],):
        raise ValueError("y must match the row count of X")
    if min_categories is not None and not isinstance(min_categories, int) and min_categories.shape != (X.shape[1],):
        raise ValueError("min_categories must match feature count")
    if sample_weight is not None and sample_weight.shape != (X.shape[0],):
        raise ValueError("sample_weight must match the row count of X")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_categorical_nb_feature_log_prob(
    category_count: list[AbstractArray],
    *,
    alpha: float = 1.0,
) -> AbstractArray:
    """Describe smoothed categorical feature log probabilities."""
    if not category_count:
        raise ValueError("category_count must be nonempty")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return AbstractArray(shape=(len(category_count),), dtype="float64")


def witness_categorical_nb_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    alpha: float = 1.0,
    fit_prior: bool = True,
    class_prior: AbstractArray | None = None,
    min_categories: AbstractArray | int | None = None,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe dense categorical naive Bayes state learned from labels."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if y.shape != (X.shape[0],):
        raise ValueError("y must match the row count of X")
    if class_prior is not None and len(class_prior.shape) != 1:
        raise ValueError("class_prior must be 1D")
    if min_categories is not None and not isinstance(min_categories, int) and min_categories.shape != (X.shape[1],):
        raise ValueError("min_categories must match feature count")
    if sample_weight is not None and sample_weight.shape != (X.shape[0],):
        raise ValueError("sample_weight must match the row count of X")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_categorical_nb_joint_log_likelihood(X: AbstractArray, state: CategoricalNBState) -> AbstractArray:
    """Describe categorical class joint log likelihoods for each row."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]), int(state.classes.shape[0])), dtype="float64")


def witness_categorical_nb_predict_log_proba(X: AbstractArray, state: CategoricalNBState) -> AbstractArray:
    """Describe normalized categorical log probabilities for each class."""
    return witness_categorical_nb_joint_log_likelihood(X, state)


def witness_categorical_nb_predict_proba(X: AbstractArray, state: CategoricalNBState) -> AbstractArray:
    """Describe normalized categorical probabilities for each class."""
    return witness_categorical_nb_joint_log_likelihood(X, state)


def witness_categorical_nb_predict(X: AbstractArray, state: CategoricalNBState) -> AbstractArray:
    """Describe one integer class prediction per categorical input row."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="int64")
