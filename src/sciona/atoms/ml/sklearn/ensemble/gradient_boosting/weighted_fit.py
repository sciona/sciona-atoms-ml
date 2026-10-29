"""A single, domain-independent, weighted gradient-boosting classifier fit."""
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier
from threadpoolctl import threadpool_limits

from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_fit_weighted_gradient_boosting(
    X: AbstractArray, labels: AbstractArray, sample_weight: object = None,
    n_estimators: int = 500, max_depth: int = 9, min_samples_split: int = 8,
    learning_rate: float = .1, random_state: int = 1,
) -> object:
    if len(X.shape) != 2 or labels.shape != (X.shape[0],):
        raise ValueError('aligned matrix and label vector required')
    return {'type': 'GradientBoostingClassifier', 'n_features': X.shape[1]}


@register_atom(witness_fit_weighted_gradient_boosting)
def fit_weighted_gradient_boosting(
    X: NDArray[np.float64], labels: NDArray[np.int64], sample_weight: object = None,
    n_estimators: int = 500, max_depth: int = 9, min_samples_split: int = 8,
    learning_rate: float = .1, random_state: int = 1,
) -> object:
    """Fit one in-memory classifier with explicit labels and optional weights.

    Accepts a finite nonempty real matrix and an aligned integer label vector
    containing at least two classes. Labels are arbitrary integers; there are no
    causal roles, paired-row assumptions, feature recipes or balancing rules.
    Optional weights must be finite, strictly positive and aligned. None is
    passed through unchanged. Uses sklearn log_loss and explicit deterministic
    settings; native thread pools are limited to one for this fit. Returns the
    fitted model without reading or writing files or mutating caller arrays.
    Zero/negative weights, missing values and single-class fits are outside this
    version's contract. No prediction quality or historical-library claim.
    """
    values = np.asarray(X)
    targets = np.asarray(labels)
    if (values.dtype.kind not in 'iuf' or values.ndim != 2 or
            min(values.shape) == 0 or not np.all(np.isfinite(values))):
        raise ValueError('finite nonempty real feature matrix required')
    if (targets.dtype.kind not in 'iu' or targets.shape != (len(values),) or
            len(np.unique(targets)) < 2):
        raise ValueError('aligned integer labels with at least two classes required')
    weight = None
    if sample_weight is not None:
        weight = np.asarray(sample_weight)
        if (weight.dtype.kind not in 'iuf' or weight.shape != (len(values),) or
                not np.all(np.isfinite(weight)) or np.any(weight <= 0)):
            raise ValueError('finite strictly positive aligned weights required')
        weight = weight.astype(np.float64, copy=True)
    for name, value, minimum in [('n_estimators', n_estimators, 1),
                                 ('max_depth', max_depth, 1),
                                 ('min_samples_split', min_samples_split, 2)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(name + ' outside integer domain')
    if (isinstance(random_state, (bool, np.bool_)) or
            not isinstance(random_state, (int, np.integer)) or not 0 <= random_state < 2**32):
        raise ValueError('unsigned 32-bit random seed required')
    if (isinstance(learning_rate, (bool, np.bool_)) or
            not isinstance(learning_rate, (int, float, np.integer, np.floating)) or
            not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('finite positive learning rate required')
    model = GradientBoostingClassifier(
        loss='log_loss', n_estimators=int(n_estimators), max_depth=int(max_depth),
        min_samples_split=int(min_samples_split), learning_rate=float(learning_rate),
        random_state=int(random_state),
    )
    with threadpool_limits(limits=1):
        model.fit(values.copy(), targets.copy(), sample_weight=weight)
    return model
