"""Draft Community boundaries for generic image-tabular classification."""
from sciona.ghost.registry import register_atom


def witness_image_tabular_prepare(payload: dict) -> dict:return {'kind':'ImageTabular.Prepared'}


@register_atom(witness_image_tabular_prepare)
def image_tabular_prepare(payload: dict) -> object:
    """Validate disjoint image/metadata populations and grouped folds."""
    from sciona.image_tabular_contract import prepare
    return prepare(payload)


def witness_image_tabular_execute(prepared: object) -> dict:
    if prepared!={'kind':'ImageTabular.Prepared'}:raise ValueError('Prepared tabular input required')
    return {'kind':'ImageTabular.Result'}


@register_atom(witness_image_tabular_execute)
def image_tabular_execute(prepared: object) -> dict:
    """Fit fold-local fusion MLPs, calibrate threshold and predict."""
    from sciona.image_tabular_contract import execute
    return execute(prepared)
