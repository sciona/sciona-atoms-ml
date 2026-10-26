"""Draft source-policy adapters and reusable averaging for meta ensembles.

Witnesses express in-process stage kinds only. Runtime numerical validation is
mandatory; these witnesses do not establish static shape or training validity.
No catalog approval is implied by registration.
"""
from sciona.ghost.registry import register_atom


def witness_prepare(layout: dict, labels: object, folds: object, identities: list,
                    seed: int, candidates: dict) -> dict:
    return {'kind': 'Otto.MetaInputs'}


@register_atom(witness_prepare)
def prepare_meta(layout: dict, labels: object, folds: object, identities: list,
                 seed: int, candidates: dict) -> object:
    from sciona.stacking_meta_operations import prepare_meta_inputs
    return prepare_meta_inputs(layout, labels, folds, identities, seed, candidates)


def witness_select(inputs: dict) -> dict:
    if inputs != {'kind': 'Otto.MetaInputs'}:
        raise ValueError('Prepared meta input required')
    return {'kind': 'Otto.SelectedControls'}


@register_atom(witness_select)
def select_xgboost(inputs: object) -> object:
    from sciona.stacking_meta_operations import select_family
    return select_family(inputs, 'xgboost')


@register_atom(witness_select)
def select_neural(inputs: object) -> object:
    from sciona.stacking_meta_operations import select_family
    return select_family(inputs, 'lasagne_neural')


@register_atom(witness_select)
def select_extratrees(inputs: object) -> object:
    from sciona.stacking_meta_operations import select_family
    return select_family(inputs, 'adaboost_extratrees')


def witness_refit(inputs: dict, selection: dict) -> dict:
    if inputs != {'kind': 'Otto.MetaInputs'} or selection != {'kind': 'Otto.SelectedControls'}:
        raise ValueError('Prepared inputs and selection required')
    return {'kind': 'Categorical.ProbabilityBag'}


@register_atom(witness_refit)
def refit_meta(inputs: object, selection: object) -> object:
    from sciona.stacking_meta_operations import refit_family
    return refit_family(inputs, selection)


def witness_mean(values: dict) -> dict:
    if values != {'kind': 'Categorical.ProbabilityBag'}:
        raise ValueError('Categorical probability bag required')
    return {'kind': 'Categorical.Probabilities'}


@register_atom(witness_mean)
def mean_probability_bag(values: object) -> object:
    """Average runs for any aligned categorical class count of at least two."""
    from sciona.probability_ensemble import mean_probability_bag as mean
    return mean(values)


def witness_blend(xgboost: dict, neural: dict, extratrees: dict) -> dict:
    if any(x != {'kind': 'Categorical.Probabilities'} for x in (xgboost, neural, extratrees)):
        raise ValueError('Aligned categorical probability matrices required')
    return {'kind': 'Otto.Blend'}


@register_atom(witness_blend)
def blend_meta(xgboost: object, neural: object, extratrees: object) -> dict:
    """Apply the fixed source mixture after per-family means, then normalize."""
    import numpy as np
    from sciona.probability_ensemble import geometric_arithmetic_probability_blend
    if any(np.asarray(x).ndim != 2 or np.asarray(x).shape[1] != 9
           for x in (xgboost, neural, extratrees)):
        raise ValueError('Source blend requires nine classes')
    return geometric_arithmetic_probability_blend(xgboost, neural, extratrees, .85, .15, .65)


def witness_result(blend: dict, xgboost: dict, neural: dict, extratrees: dict) -> dict:
    if blend != {'kind': 'Otto.Blend'} or any(
            x != {'kind': 'Otto.SelectedControls'} for x in (xgboost, neural, extratrees)):
        raise ValueError('Source blend and three selections required')
    return {'kind': 'Otto.MetaResult'}


@register_atom(witness_result)
def meta_result(blend: dict, xgboost: object, neural: object, extratrees: object) -> dict:
    from sciona.stacking_meta_operations import FAMILIES, SelectedControls, selection_summary
    selections = (xgboost, neural, extratrees)
    if any(type(s) is not SelectedControls or s.family != family
           for family, s in zip(FAMILIES, selections)):
        raise ValueError('Selection family order differs')
    if any(s.inputs is not xgboost.inputs for s in selections):
        raise ValueError('Selections must share one prepared population')
    return dict(blend=blend, selection={family: selection_summary(s)
                                       for family, s in zip(FAMILIES, selections)})
