"""Explicit row/schema boundaries for composable numerical training graphs."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from sciona.atoms.ml.xgboost.model_io import _matrix


def witness_training_inputs(features, targets, groups, feature_names):
    if len(features.shape)!=2 or features.dtype!='float64' or targets.shape!=(features.shape[0],) or targets.dtype!='float64' or groups.shape!=targets.shape or groups.dtype!='int64':
        raise ValueError('Aligned training metadata required')
    return features,targets,groups,list(feature_names)

@register_atom(witness_training_inputs)
def training_inputs(features: NDArray[np.float64], targets: NDArray[np.float64], groups: NDArray[np.int64], feature_names: list) -> tuple:
    """Validate common row alignment and ordered schema; expose independent graph ports."""
    _matrix(features,feature_names)
    y,g=np.asarray(targets),np.asarray(groups)
    if y.dtype!=np.float64 or g.dtype!=np.int64 or y.shape!=(len(features),) or g.shape!=y.shape or not np.isfinite(y).all():
        raise ValueError('Aligned finite float64 targets and int64 groups required')
    return features,targets,groups,list(feature_names)


def witness_prediction_inputs(prediction_features,feature_names):
    if len(prediction_features.shape)!=2 or prediction_features.dtype!='float64' or prediction_features.shape[1]!=len(feature_names):
        raise ValueError('Ordered prediction matrix metadata required')
    return prediction_features

@register_atom(witness_prediction_inputs)
def prediction_inputs(prediction_features: NDArray[np.float64], feature_names: list) -> NDArray[np.float64]:
    """Validate a separate prediction population against the supplied ordered schema."""
    _matrix(prediction_features,feature_names)
    return prediction_features


def witness_append_feature_name(feature_names,prediction_feature_name):
    return append_feature_name(feature_names,prediction_feature_name)

@register_atom(witness_append_feature_name)
def append_feature_name(feature_names: list, prediction_feature_name: str) -> list:
    """Append one unique explicit name; preserve the existing feature order."""
    if not isinstance(feature_names,list) or any(not isinstance(n,str) or not n for n in feature_names) or len(set(feature_names))!=len(feature_names):
        raise ValueError('Unique ordered feature names required')
    if not isinstance(prediction_feature_name,str) or not prediction_feature_name or prediction_feature_name in feature_names:
        raise ValueError('Unique new prediction feature name required')
    return feature_names+[prediction_feature_name]
