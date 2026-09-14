"""Source Feng boosted-tree segment prediction on ordered FFT tensors."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_feng_xgb_segment_probabilities(training_features: AbstractArray, segment_labels: AbstractArray,
                                          prediction_features: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=(prediction_features.shape[0],), dtype='float32')


@register_atom(witness_feng_xgb_segment_probabilities)
def feng_xgb_segment_probabilities(training_features: NDArray[np.float64], segment_labels: NDArray[np.int64],
                                   prediction_features: NDArray[np.float64]) -> NDArray[np.float32]:
    """Fit 500 source-configured boosting rounds and mean window probabilities.

    Input axes are segments/channels/features/windows. Round features through
    source float32 loading, then reshape into float64 segment/window rows with
    channel/feature columns. No scaling or KNN nonfinite cleanup is applied.
    NaN is accepted as XGBoost missing data; both signs of infinity are rejected.
    Both binary classes are required, one label per training segment. Channel and
    feature axes must match between partitions; window counts may differ.

    Source params: binary:logistic, gbtree, AUC, eta .22, depth3, subsample .80,
    colsample_bytree .78, 500 rounds. Current XGBoost defaults apply to unspecified
    parameters; legacy silent=1 maps to verbosity=0. Serial CPU execution. This
    does not reproduce the historical 2016 engine or claim predictive validity.
    Output is mean positive-class probability per segment in prediction order.
    """
    import xgboost as xgb
    train, prediction = np.asarray(training_features), np.asarray(prediction_features)
    labels = np.asarray(segment_labels)
    for value in [train, prediction]:
        if value.dtype.kind not in 'iuf' or value.ndim != 4 or min(value.shape) == 0 or np.any(np.isinf(value)):
            raise ValueError('nonempty real segment/channel/feature/window tensors without infinities required')
    if train.shape[1:3] != prediction.shape[1:3]:
        raise ValueError('matching channel and feature axes required')
    if labels.dtype.kind not in 'iuf' or labels.shape != (len(train),) or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError('one binary label per training segment with both classes required')
    def rows(value):
        with np.errstate(over='raise', invalid='raise'):
            rounded = value.astype(np.float32)
        return rounded.transpose(0, 3, 1, 2).reshape(-1, value.shape[1]*value.shape[2]).astype(np.float64)
    x, xp = rows(train), rows(prediction)
    params = dict(objective='binary:logistic', booster='gbtree', eval_metric='auc', eta=.22,
                  max_depth=3, subsample=.80, colsample_bytree=.78, verbosity=0, nthread=1)
    model = xgb.train(params, xgb.DMatrix(x, label=np.repeat(labels, train.shape[3]), nthread=1),
                      num_boost_round=500, verbose_eval=False)
    probabilities = model.predict(xgb.DMatrix(xp, nthread=1))
    result = probabilities.reshape(len(prediction), prediction.shape[3]).mean(axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite source XGBoost predictions')
    return result
