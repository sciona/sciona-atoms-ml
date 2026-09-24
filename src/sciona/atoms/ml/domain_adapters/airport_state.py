"""Draft state boundaries for separate airport training and inference graphs."""
import numpy as np

from sciona.ghost.registry import register_atom
from sciona.airport_model_state import validate_adapter_state
from sciona.residual_model_state import pack_state
from sciona.atoms.ml.domain_adapters.airport_features import (
    numerical_handoff,witness_numerical_handoff,witness_unmaterialized,
)


def witness_training_handoff(features: object,targets: object,groups: object,feature_names: list,
        train_fraction: object,seed: object,maximum_error: object,threshold: object,prediction_feature_name: str) -> tuple:
    values=witness_numerical_handoff(features,targets,groups,feature_names,features,
        train_fraction,seed,maximum_error,threshold,prediction_feature_name)
    return values[:4]+values[5:]


@register_atom(witness=witness_training_handoff)
def training_handoff(features: np.ndarray,targets: np.ndarray,groups: np.ndarray,feature_names: list,
        train_fraction: float,seed: int,maximum_error: float,threshold: float,prediction_feature_name: str) -> tuple:
    """Validate training metadata without receiving future query records.

    The observed training matrix supplies a shape probe for the numerical
    witness traversal. This does not qualify an unseen query population.
    """
    values=numerical_handoff(features,targets,groups,feature_names,features,
        train_fraction,seed,maximum_error,threshold,prediction_feature_name)
    return values[:4]+values[5:]


@register_atom(witness=witness_unmaterialized)
def pack_model_state(feature_names: list,prediction_feature_name: str,regressor: dict,classifier: dict,
        offsets: np.ndarray,threshold: float,vocabulary: tuple,airport: str) -> dict:
    state=pack_state(feature_names,prediction_feature_name,regressor,classifier,offsets,threshold,
        dict(format='sciona.airport.feature-state.v1',airport=airport,vocabulary=list(vocabulary),target_unit='minutes'))
    validate_adapter_state(state)
    return state


@register_atom(witness=witness_unmaterialized)
def unpack_model_state(model_state: dict) -> tuple:
    adapter,names,vocabulary=validate_adapter_state(model_state)
    return (adapter['airport'],names,vocabulary,model_state['regressor'],model_state['classifier'],
            np.asarray(model_state['offsets'],dtype=np.float64),model_state['threshold'],
            model_state['prediction_feature_name'],names+[model_state['prediction_feature_name']])
