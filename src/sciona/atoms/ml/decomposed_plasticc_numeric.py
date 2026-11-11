"""Reusable numeric operations extracted during PLAsTiCC qualification.

Local registration does not imply catalog approval. Astronomy policy is held
in separate lifecycle adapters; all controls here are explicit caller inputs.
"""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete complete inputs required')


@register_atom(concrete_only)
def initial_amplitude(targets: np.ndarray, errors: np.ndarray, relative_floor: float, amplitude_fraction: float) -> float:
    """Estimate covariance amplitude from explicitly noise-adjusted responses."""
    from sciona._runtime.ml.decomposed_plasticc_gp_fit import initial_amplitude as run
    return run(targets, errors, relative_floor, amplitude_fraction)


@register_atom(concrete_only)
def fit_gp(coordinates: np.ndarray, targets: np.ndarray, errors: np.ndarray, amplitude: float, lengths: np.ndarray, optimized_axis: int, metric_log_bounds: np.ndarray, amplitude_log_radius: float, fix_amplitude: bool, white_noise_variance: float) -> object:
    """Fit one coordinate length and optionally amplitude with explicit bounds."""
    from sciona._runtime.ml.decomposed_plasticc_gp_fit import fit as run
    return run(coordinates, targets, errors, amplitude, lengths, optimized_axis, metric_log_bounds, amplitude_log_radius, fix_amplitude, white_noise_variance)


@register_atom(concrete_only)
def scale_covariance(normalized: np.ndarray, amplitude: float) -> np.ndarray:
    """Multiply a covariance block by an explicit positive variance."""
    from sciona._runtime.ml.decomposed_plasticc_gp import scale_covariance as run
    return run(normalized, amplitude)


@register_atom(concrete_only)
def observation_variance(errors: np.ndarray, white_noise_variance: np.ndarray) -> np.ndarray:
    """Combine aligned errors and explicit white-noise variance."""
    from sciona._runtime.ml.decomposed_plasticc_gp import observation_variance as run
    return run(errors, white_noise_variance)


@register_atom(concrete_only)
def condition(covariance: np.ndarray, targets: np.ndarray, noise_variance: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Condition a zero-mean GP without jitter or singular-matrix repair."""
    from sciona._runtime.ml.decomposed_plasticc_gp import condition as run
    return run(covariance, targets, noise_variance)


@register_atom(concrete_only)
def predict_gp(factor: np.ndarray, alpha: np.ndarray, cross_covariance: np.ndarray, prior_variance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return conditional means and unclipped latent marginal variances."""
    from sciona._runtime.ml.decomposed_plasticc_gp import predict as run
    return run(factor, alpha, cross_covariance, prior_variance)


@register_atom(concrete_only)
def balanced_weights(labels: object, class_weights: object) -> object:
    """Balance indexed class labels with optional explicit class weights."""
    from sciona._runtime.ml.decomposed_plasticc_sampling import balanced_weights as run
    return run(labels, class_weights)


@register_atom(concrete_only)
def reference_stratified_folds(labels: object, references: object, num_folds: int, seed: int) -> object:
    """Stratify originals and keep each derivative in its original's fold."""
    from sciona._runtime.ml.decomposed_plasticc_sampling import reference_stratified_folds as run
    return run(labels, references, num_folds, seed)


@register_atom(concrete_only)
def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Apply retained unshifted exponential row normalization."""
    from sciona._runtime.ml.decomposed_plasticc_prediction import normalize_scores as run
    return run(scores)


@register_atom(concrete_only)
def predict_scores(model: object, features: object, classes: object) -> np.ndarray:
    """Predict raw scores at the selected iteration with checked class order."""
    from sciona._runtime.ml.decomposed_plasticc_prediction import predict_scores as run
    return run(model, features, classes)


@register_atom(concrete_only)
def average_folds(models: list, features: object, classes: object, score_operation: object, normalize_operation: object) -> object:
    """Average every fold in order through explicit scoring callables."""
    from sciona._runtime.ml.decomposed_plasticc_prediction import average_folds as run
    return run(models, features, classes, score_operation, normalize_operation)
