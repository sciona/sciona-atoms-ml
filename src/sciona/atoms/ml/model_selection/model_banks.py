"""Materialized domain-neutral model-slot composition and selection."""
from sciona.ghost.registry import register_atom
from sciona.model_slot_bank import assemble_model_slot_bank, select_model_slots


def witness_materialized(*args: object, **kwargs: object) -> object:
    raise ValueError('Materialized model mappings required; static propagation is unsupported')


@register_atom(witness=witness_materialized)
def assemble_bank(models: dict, population_bindings: dict, shared_bindings: dict) -> dict:
    """Resolve explicit local aliases and shared slots without copying model objects."""
    return assemble_model_slot_bank(models, population_bindings, shared_bindings)


@register_atom(witness=witness_materialized)
def select_slots(bank: dict, population: str) -> dict:
    """Select one named population into an owned map, rejecting unknown populations."""
    return select_model_slots(bank, population)
