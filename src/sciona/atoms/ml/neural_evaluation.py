"""Selected neural prediction and diagnostic loss without fitting feedback."""
import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits
from sciona.ghost.abstract import AbstractArray,AbstractScalar
from sciona.ghost.registry import register_atom
from .numpy_adagrad_binary import predict_adagrad_binary,witness_predict_adagrad_binary,_parameters,_matrix,_loss_gradients
from .model_selection.masked_operations import _mask,_mask_metadata,_selected


def witness_masked_neural_predictions(features: AbstractArray,feature_names: list,state: dict,selection: AbstractArray) -> AbstractArray:
    result=witness_predict_adagrad_binary(features,feature_names,state)
    _mask_metadata(selection,features.shape[0])
    return result


@register_atom(witness_masked_neural_predictions)
def masked_neural_predictions(features: NDArray[np.float64],feature_names: list,state: dict,selection: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Predict selected rows only, scattering into an otherwise zero vector.

    Selection is nonempty and explicit. Excluded feature values are not read
    by the predictor. Zeros are placeholders, not predictions: downstream
    contribution accounting must use the same selection mask.
    """
    x=np.asarray(features)
    if x.ndim!=2:raise ValueError('Feature matrix required')
    mask=_mask(selection,len(x));result=np.zeros(len(x),dtype=np.float64)
    result[mask]=predict_adagrad_binary(x[mask],feature_names,state)
    return result


def witness_masked_neural_loss(features: AbstractArray,targets: AbstractArray,feature_names: list,state: dict,selection: AbstractArray) -> AbstractScalar:
    witness_masked_neural_predictions(features,feature_names,state,selection)
    if targets.shape!=(features.shape[0],) or targets.dtype!='int64':raise ValueError('Aligned int64 targets required')
    if targets.dim is not None and not targets.dim.is_dimensionless:raise ValueError('Dimensionless labels required')
    return AbstractScalar(dtype='float64',min_val=0)


@register_atom(witness_masked_neural_loss)
def masked_neural_loss(features: NDArray[np.float64],targets: NDArray[np.int64],feature_names: list,state: dict,selection: NDArray[np.bool_]) -> float:
    """Mean stable softmax cross entropy on selected rows, diagnostic only.

    No optimizer update or early-stopping decision. One class is permitted in
    the diagnostic selection; labels must be int64 zero/one. Excluded labels
    and feature values cannot influence the loss.
    """
    x,y=_selected(features,targets,selection);x=_matrix(x,feature_names)
    if y.dtype!=np.int64 or not np.isin(y,[0,1]).all():raise ValueError('Selected binary int64 targets required')
    parameters=_parameters(state,feature_names)
    with threadpool_limits(limits=1),np.errstate(over='raise',invalid='raise',divide='raise'):
        loss,_=_loss_gradients(x,y,parameters)
    if not np.isfinite(loss):raise ValueError('Nonfinite validation loss')
    return float(loss)
