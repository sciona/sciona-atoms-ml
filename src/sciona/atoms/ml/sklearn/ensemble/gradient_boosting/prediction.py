"""Execute probability prediction with an explicitly supplied fitted classifier."""
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_gradient_boosting_class_probabilities(
    estimator: object, X: AbstractArray,
) -> tuple[AbstractArray, AbstractArray]:
    if len(X.shape) != 2:
        raise ValueError('feature matrix required')
    classes=len(estimator.classes_)
    return (AbstractArray(shape=(X.shape[0],classes),dtype='float64'),
            AbstractArray(shape=(classes,),dtype=str(estimator.classes_.dtype)))


@register_atom(witness_gradient_boosting_class_probabilities)
def gradient_boosting_class_probabilities(
    estimator: object, X: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray]:
    """Predict probabilities and return a copy of their corresponding class labels.

    Requires an in-memory fitted sklearn GradientBoostingClassifier and a finite
    real, nonempty feature matrix with the trained number of columns. The model
    is supplied by the caller; this function does not fit or deserialize models.
    Class order comes from classes_, never an assumed ordering. Prediction
    semantics follow sklearn's public predict_proba API.
    """
    if not isinstance(estimator,GradientBoostingClassifier):
        raise TypeError('GradientBoostingClassifier required')
    check_is_fitted(estimator)
    values=np.asarray(X)
    if values.dtype.kind not in 'iuf' or values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != estimator.n_features_in_ or not np.all(np.isfinite(values)):
        raise ValueError('finite nonempty feature matrix matching fitted columns required')
    probabilities=np.asarray(estimator.predict_proba(values),dtype=np.float64)
    labels=estimator.classes_.copy()
    if probabilities.shape != (len(values),len(labels)) or not np.all(np.isfinite(probabilities)) or np.any(probabilities<0) or np.any(probabilities>1) or not np.allclose(probabilities.sum(axis=1),1.,rtol=0,atol=1e-8):
        raise ValueError('invalid classifier probability output')
    return probabilities,labels
