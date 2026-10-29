"""Explicit stratified inner partitions within an ordered outer selection."""
import numbers
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from .masked_operations import _mask,_mask_metadata


def _controls(n_splits,fold_index):
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,numbers.Integral) for v in (n_splits,fold_index)):
        raise ValueError('Integer split count and fold index required')
    if n_splits<2 or not 0<=fold_index<n_splits:
        raise ValueError('At least two splits and an in-range fold index required')


def witness_stratified_inner_masks(labels: AbstractArray,outer_selection: AbstractArray,
        n_splits: int,fold_index: int) -> tuple:
    if len(labels.shape)!=1 or labels.shape[0]<2 or labels.dtype!='int64':
        raise ValueError('Nonempty int64 class-label vector required')
    if labels.dim is not None and not labels.dim.is_dimensionless:
        raise ValueError('Class labels are dimensionless')
    _mask_metadata(outer_selection,labels.shape[0]);_controls(n_splits,fold_index)
    return (AbstractArray(shape=labels.shape,dtype='bool'),AbstractArray(shape=labels.shape,dtype='bool'))


@register_atom(witness_stratified_inner_masks)
def stratified_inner_masks(labels: NDArray[np.int64],outer_selection: NDArray[np.bool_],
        n_splits: int,fold_index: int) -> tuple[NDArray[np.bool_],NDArray[np.bool_]]:
    """Return inner fit/validation masks in the original full row coordinates.

    Uses current sklearn StratifiedKFold without shuffling on selected rows in
    their original order. Each selected class needs at least n_splits rows;
    at least two classes are required. Outer-excluded label values are never
    examined for class membership. Outputs are disjoint and their union is
    exactly the outer selection. No identity/group independence is inferred.
    """
    from sklearn.model_selection import StratifiedKFold
    y=np.asarray(labels)
    if y.dtype!=np.int64 or y.ndim!=1 or len(y)<2:
        raise ValueError('Nonempty int64 class-label vector required')
    mask=_mask(outer_selection,len(y));_controls(n_splits,fold_index)
    indices=np.flatnonzero(mask);selected=y[indices]
    classes,counts=np.unique(selected,return_counts=True)
    if len(classes)<2 or counts.min()<n_splits:
        raise ValueError('At least two selected classes with n_splits rows each required')
    splitter=StratifiedKFold(int(n_splits),shuffle=False)
    for index,(fit,validation) in enumerate(splitter.split(np.zeros((len(selected),1)),selected)):
        if index==fold_index:break
    fit_mask=np.zeros(len(y),dtype=np.bool_);validation_mask=np.zeros(len(y),dtype=np.bool_)
    fit_mask[indices[fit]]=True;validation_mask[indices[validation]]=True
    return fit_mask,validation_mask
