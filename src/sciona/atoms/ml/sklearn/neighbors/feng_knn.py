"""Source-configured Feng KNN training and segment scoring."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_feng_knn_segment_probabilities(training_features: AbstractArray, segment_labels: AbstractArray,
                                          prediction_features: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(prediction_features.shape[0],), dtype='float64')


@register_atom(witness_feng_knn_segment_probabilities)
def feng_knn_segment_probabilities(training_features: NDArray[np.float64], segment_labels: NDArray[np.int64],
                                   prediction_features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Train 40-neighbor distance-weighted Manhattan KNN and average window scores.

    Feature tensors have axes segments/channels/features/windows, with matching
    channel/feature axes. Window counts may differ between train and prediction;
    each partition has a rectangular tensor. Labels contain both 0 and 1, one
    per training segment. At least 40 training windows are required.

    Source file loading rounds features to float32, then reshape expands into
    float64 rows ordered segment/window with channel/feature columns. Negative
    infinity and NaN are replaced with zero; positive infinity is rejected.
    Each partition fits its OWN StandardScaler, as in the public source. Thus
    prediction depends on the entire prediction batch. This unusual behavior
    is deliberate source fidelity. Returns mean positive-class probability per
    prediction segment in input order, not maximum probability. Uses current
    sklearn with serial execution; no historical-engine or clinical claim.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    train, prediction = np.asarray(training_features), np.asarray(prediction_features)
    labels = np.asarray(segment_labels)
    for value in [train, prediction]:
        if value.dtype.kind not in 'iuf' or value.ndim != 4 or min(value.shape) == 0 or np.any(np.isposinf(value)):
            raise ValueError('nonempty real segment/channel/feature/window tensors without positive infinity required')
    if train.shape[1:3] != prediction.shape[1:3] or train.shape[0] * train.shape[3] < 40:
        raise ValueError('matching channel/feature axes and at least 40 training windows required')
    if labels.dtype.kind not in 'iuf' or labels.shape != (len(train),) or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError('one binary label per training segment, with both classes required')

    def rows(value):
        with np.errstate(over='raise', invalid='raise'):
            rounded = value.astype(np.float32)
        matrix = rounded.transpose(0, 3, 1, 2).reshape(-1, value.shape[1]*value.shape[2]).astype(np.float64)
        matrix[np.isneginf(matrix) | np.isnan(matrix)] = 0
        return StandardScaler().fit_transform(matrix)

    x, xp = rows(train), rows(prediction)
    model = KNeighborsClassifier(n_neighbors=40, weights='distance', metric='manhattan', n_jobs=1)
    model.fit(x, np.repeat(labels, train.shape[3]))
    probabilities = model.predict_proba(xp)[:, 1]
    result = probabilities.reshape(len(prediction), prediction.shape[3]).mean(axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite source KNN predictions')
    return result
