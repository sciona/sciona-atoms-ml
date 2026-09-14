"""Execute the five trained classifiers of the three-system causal ensemble."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from sciona.atoms.causal_inference.estimators.probabilities import binary_positive_class_probability
from .prediction import gradient_boosting_class_probabilities


def witness_causal_classifier_predictions(X: AbstractArray, one_step_model: object,
    independence_model: object, direction_model: object, left_model: object,
    right_model: object) -> tuple:
    if len(X.shape)!=2:raise ValueError('feature matrix required')
    vector=AbstractArray(shape=(X.shape[0],),dtype='float64')
    return (AbstractArray(shape=(X.shape[0],3),dtype='float64'),AbstractArray(shape=(3,),dtype='int64'),vector,vector,vector,vector)


@register_atom(witness_causal_classifier_predictions)
def causal_classifier_predictions(X: NDArray[np.float64], one_step_model: object,
    independence_model: object, direction_model: object, left_model: object,
    right_model: object) -> tuple:
    """Predict all five classifiers on a shared, ordered feature matrix.

    Models are fitted GradientBoostingClassifiers supplied in memory. One-step
    labels must be {-1,0,1}; dependence and left/right indicators use {0,1};
    direction uses {-1,1}. Binary outputs are P(label=1), including direction,
    matching the original estimator's probability convention. This function
    does not train models or claim that their features, labels or predictions
    establish causal validity. Rows must retain the same orientation ordering
    across all models; pairing and ensemble combination occur downstream.
    """
    models=[one_step_model,independence_model,direction_model,left_model,right_model]
    required=[{-1,0,1},{0,1},{-1,1},{0,1},{0,1}]
    predictions=[]
    for model,labels in zip(models,required):
        p,c=gradient_boosting_class_probabilities(model,X)
        if c.dtype.kind not in 'iu' or set(c.tolist())!=labels:
            raise ValueError('classifier labels do not match the declared causal role')
        predictions.append((p,c))
    p,c=predictions[0]
    return (p,c,*(binary_positive_class_probability(a,b) for a,b in predictions[1:]))
