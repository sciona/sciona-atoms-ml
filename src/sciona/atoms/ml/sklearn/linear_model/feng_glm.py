"""Source-configured Feng logistic regression training and segment scoring."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_feng_glm_segment_probabilities(training_features: AbstractArray, segment_labels: AbstractArray,
                                          prediction_features: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(prediction_features.shape[0],), dtype='float64')


@register_atom(witness_feng_glm_segment_probabilities)
def feng_glm_segment_probabilities(training_features: NDArray[np.float64], segment_labels: NDArray[np.int64],
                                   prediction_features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fit source L2/C=.6 logistic regression and mean window probabilities.

    Inputs are segments/windows/features tensors with matching feature width and
    both binary classes, one label per training segment. Source float32 loading
    and reshape precede scaling. Only negative infinity is zeroed; NaNs and
    positive infinity are rejected. Prediction reuses the training scaler.
    Mean positive-class window scores preserve prediction segment order.

    Uses current sklearn's LBFGS default, max_iter100, tol1e-4, intercept and L2
    penalty with C=.6. Source leaves solver unspecified; this is explicitly a
    current-library source execution, not historical 2016 solver reproduction.
    Nonconvergence is rejected rather than silently returning an unfinished fit.
    No clinical or predictive-performance claim. No input mutation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    from sklearn.preprocessing import StandardScaler

    train, prediction = np.asarray(training_features), np.asarray(prediction_features)
    labels = np.asarray(segment_labels)
    for value in [train, prediction]:
        if value.dtype.kind not in 'iuf' or value.ndim != 3 or min(value.shape) == 0 or np.any(np.isposinf(value)) or np.any(np.isnan(value)):
            raise ValueError('nonempty real segment/window/feature tensors without NaN or positive infinity required')
    if train.shape[2] != prediction.shape[2]:
        raise ValueError('matching feature widths required')
    if labels.dtype.kind not in 'iuf' or labels.shape != (len(train),) or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError('one binary label per training segment, with both classes required')

    def rows(value):
        with np.errstate(over='raise', invalid='raise'):
            rounded = value.astype(np.float32)
        matrix = rounded.reshape(-1, value.shape[2])
        matrix[np.isneginf(matrix)] = 0
        return matrix

    scaler = StandardScaler()
    x, xp = scaler.fit_transform(rows(train)), scaler.transform(rows(prediction))
    model = LogisticRegression(C=.6, solver='lbfgs', max_iter=100, tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, np.repeat(labels, train.shape[1]))
    probabilities = model.predict_proba(xp)[:, 1]
    result = probabilities.reshape(len(prediction), prediction.shape[1]).mean(axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite source logistic predictions')
    return result
