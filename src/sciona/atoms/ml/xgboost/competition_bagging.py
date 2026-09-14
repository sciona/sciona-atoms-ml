"""Source-configured window classification with modern XGBoost and sklearn."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_bagged_window_probabilities(training_features: AbstractArray, training_segment_indices: AbstractArray,
                                        segment_labels: AbstractArray, prediction_features: AbstractArray,
                                        n_bags: int, n_estimators: int = 500) -> AbstractArray:
    return AbstractArray(shape=(prediction_features.shape[0],), dtype='float64')


@register_atom(witness_bagged_window_probabilities)
def bagged_window_probabilities(training_features: NDArray[np.float64], training_segment_indices: NDArray[np.int64],
                                segment_labels: NDArray[np.int64], prediction_features: NDArray[np.float64],
                                n_bags: int, n_estimators: int = 500) -> NDArray[np.float64]:
    """Fit source-configured bagged boosted trees and return positive-class scores.

    Feature rows correspond to windows. Explicit zero-based segment indices map
    every training row to its binary segment label; every segment must occur.
    Prediction features must have matching columns. All inputs must be finite.
    Source branches specify four bags (autocorrelation) or ten (coherence), each
    with 500 boosted trees. Other positive counts are supported explicitly.

    Preserves the public model YAML hyperparameters and sklearn row/column bagging.
    Runs serially for reproducibility. Uses current sklearn's estimator argument
    in place of base_estimator and current XGBoost defaults for unspecified values.
    This is a modern execution of the source configuration, not a reproduction of
    the original 2016 training engine. Requires the provider's xgboost extra and
    an OpenMP runtime on macOS. Models stay in memory and are not deserialized.
    """
    from sklearn.ensemble import BaggingClassifier
    from xgboost import XGBClassifier

    train, prediction = np.asarray(training_features), np.asarray(prediction_features)
    indices, labels = np.asarray(training_segment_indices), np.asarray(segment_labels)
    for value in [train, prediction]:
        if value.dtype.kind not in 'iuf' or value.ndim != 2 or min(value.shape) < 1 or not np.all(np.isfinite(value)):
            raise ValueError('finite nonempty feature matrices required')
    if train.shape[1] != prediction.shape[1] or train.shape[1] < 2 or train.shape[0] < 3:
        raise ValueError('matching feature columns, at least two columns and three training rows required')
    if labels.ndim != 1 or labels.dtype.kind not in 'iuf' or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError('segment labels must contain both binary classes 0 and 1')
    if indices.dtype.kind not in 'iu' or indices.shape != (len(train),) or not np.array_equal(np.unique(indices), np.arange(len(labels))):
        raise ValueError('training indices must cover every segment exactly by zero-based identity')
    for count in [n_bags, n_estimators]:
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError('positive integer bag/tree counts required')
    estimator = XGBClassifier(n_estimators=int(n_estimators), learning_rate=.01,
                              max_depth=4, subsample=.50, colsample_bytree=.50,
                              colsample_bylevel=1., min_child_weight=2, seed=42, n_jobs=1)
    model = BaggingClassifier(estimator=estimator, max_samples=.99, max_features=.99,
                              random_state=666, n_estimators=int(n_bags), n_jobs=1)
    model.fit(train, labels[indices].astype(int))
    positive = np.flatnonzero(model.classes_ == 1)
    if len(positive) != 1:
        raise ValueError('positive class missing from fitted model')
    probabilities = model.predict_proba(prediction)[:, positive[0]].astype(float)
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('invalid classifier probabilities')
    return probabilities
