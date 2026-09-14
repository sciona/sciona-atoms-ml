"""Source-configured Feng KNN training and segment scoring."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_feng_expanded_knn_segment_probabilities(training_features: AbstractArray, segment_labels: AbstractArray,
                                          prediction_features: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(prediction_features.shape[0],), dtype='float64')


@register_atom(witness_feng_expanded_knn_segment_probabilities)
def feng_expanded_knn_segment_probabilities(training_features: NDArray[np.float64], segment_labels: NDArray[np.int64],
                                   prediction_features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Train source expanded-feature KNN and return mean segment probabilities.

    Inputs are segments/windows/features tensors with matching feature width.
    Source loading and row reshape retain float32 through StandardScaler.
    Negative infinity and NaN are zeroed; positive infinity is rejected. Each
    partition fits its own scaler, so prediction depends on its entire batch.
    Both binary classes and at least forty training windows are required.
    Uses forty distance-weighted Manhattan neighbors and mean window scores in
    segment order. Serial current sklearn; no historical-engine or performance
    claim. Distinct from the base source model's float64 row allocation.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    train, prediction = np.asarray(training_features), np.asarray(prediction_features)
    labels = np.asarray(segment_labels)
    for value in [train, prediction]:
        if value.dtype.kind not in 'iuf' or value.ndim != 3 or min(value.shape) == 0 or np.any(np.isposinf(value)):
            raise ValueError('nonempty real segment/window/feature tensors without positive infinity required')
    if train.shape[2] != prediction.shape[2] or train.shape[0] * train.shape[1] < 40:
        raise ValueError('matching feature widths and at least 40 training windows required')
    if labels.dtype.kind not in 'iuf' or labels.shape != (len(train),) or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError('one binary label per training segment, with both classes required')

    def rows(value):
        with np.errstate(over='raise', invalid='raise'):
            rounded = value.astype(np.float32)
        matrix = rounded.reshape(-1, value.shape[2])
        matrix[np.isneginf(matrix) | np.isnan(matrix)] = 0
        return StandardScaler().fit_transform(matrix)

    x, xp = rows(train), rows(prediction)
    model = KNeighborsClassifier(n_neighbors=40, weights='distance', metric='manhattan', n_jobs=1)
    model.fit(x, np.repeat(labels, train.shape[1]))
    probabilities = model.predict_proba(xp)[:, 1]
    result = probabilities.reshape(len(prediction), prediction.shape[1]).mean(axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite source KNN predictions')
    return result
