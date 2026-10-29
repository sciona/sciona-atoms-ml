"""Reusable shuffled-fold masks and configured fitting on selected rows."""
import numbers
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from .masked_operations import _selected, _mask_metadata
from sciona.atoms.ml.xgboost.configured_model_io import (
    fit_configured_regression_model,fit_configured_binary_model,
    witness_fit_configured_regression_model,witness_fit_configured_binary_model,
)


def _fold_controls(rows,n_splits,seed,fold_index):
    for name,value in [('n_splits',n_splits),('seed',seed),('fold_index',fold_index)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Integral):
            raise ValueError('Integer '+name+' required')
    if not 2<=n_splits<=rows or not 0<=fold_index<n_splits or not 0<=seed<2**32:
        raise ValueError('Valid shuffled-fold count, index and random-state seed required')


def witness_shuffled_kfold_masks(features: AbstractArray,n_splits: int,seed: int,fold_index: int) -> tuple:
    if len(features.shape)!=2 or min(features.shape)<1 or features.dtype!='float64':
        raise ValueError('Nonempty float64 feature matrix metadata required')
    _fold_controls(features.shape[0],n_splits,seed,fold_index)
    return (AbstractArray(shape=(features.shape[0],),dtype='bool'),
            AbstractArray(shape=(features.shape[0],),dtype='bool'))


@register_atom(witness_shuffled_kfold_masks)
def shuffled_kfold_masks(features: NDArray[np.float64],n_splits: int,seed: int,fold_index: int) -> tuple[NDArray[np.bool_],NDArray[np.bool_]]:
    """Return disjoint fit/held masks for one shuffled sklearn KFold partition.

    Only the feature matrix shape is read. No target labels, stratification,
    grouping, sorting or value-dependent selection is inferred. Integer seed
    fixes the current backend's shuffle; masks retain original row order. All
    rows belong to exactly one of the two masks and both masks are nonempty.
    Numeric feature validity is checked by the downstream fitting operation.
    """
    from sklearn.model_selection import KFold
    x=np.asarray(features)
    if x.dtype!=np.float64 or x.ndim!=2 or min(x.shape)<1:
        raise ValueError('Nonempty float64 feature matrix required')
    _fold_controls(len(x),n_splits,seed,fold_index)
    fit,held=list(KFold(int(n_splits),shuffle=True,random_state=int(seed)).split(x))[int(fold_index)]
    fit_mask=np.zeros(len(x),dtype=np.bool_);held_mask=np.zeros(len(x),dtype=np.bool_)
    fit_mask[fit]=True;held_mask[held]=True
    return fit_mask,held_mask


def witness_fit_masked_configured_regression(features: AbstractArray,targets: AbstractArray,feature_names: list,
        selection: AbstractArray,configuration: dict,boosting_rounds: int) -> tuple:
    state,rounds=witness_fit_configured_regression_model(features,targets,feature_names,configuration,boosting_rounds)
    _mask_metadata(selection,features.shape[0])
    return state,rounds


def witness_fit_masked_configured_binary(features: AbstractArray,targets: AbstractArray,feature_names: list,
        selection: AbstractArray,configuration: dict,boosting_rounds: int) -> tuple:
    state,rounds=witness_fit_configured_binary_model(features,targets,feature_names,configuration,boosting_rounds)
    _mask_metadata(selection,features.shape[0])
    return state,rounds


@register_atom(witness_fit_masked_configured_regression)
def fit_masked_configured_regression(features: NDArray[np.float64],targets: NDArray[np.float64],feature_names: list,
        selection: NDArray[np.bool_],configuration: dict,boosting_rounds: int) -> tuple[dict,int]:
    """Fit one configured squared-error model using only selected ordered rows.

    The nonempty boolean selection must align exactly with features/targets.
    Unselected rows and targets never enter fitting. Selected rows retain order,
    units, NaN missing features and the configured fit's full-round/one-thread
    contract. Return the existing native model-state ABI plus actual rounds.
    """
    x,y=_selected(features,targets,selection)
    return fit_configured_regression_model(x,y,feature_names,configuration,boosting_rounds)


@register_atom(witness_fit_masked_configured_binary)
def fit_masked_configured_binary(features: NDArray[np.float64],targets: NDArray[np.int64],feature_names: list,
        selection: NDArray[np.bool_],configuration: dict,boosting_rounds: int) -> tuple[dict,int]:
    """Fit one configured logistic model with both classes in selected rows.

    Uses the masked regression boundary's alignment and selection semantics.
    Labels outside the selected training partition cannot affect the fit.
    Graphs own held-out prediction collection, full refits and aggregation.
    """
    x,y=_selected(features,targets,selection)
    return fit_configured_binary_model(x,y,feature_names,configuration,boosting_rounds)
