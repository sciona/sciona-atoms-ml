"""Portable native regression/model-bank state with explicit integrity checks."""
from sciona.ghost.registry import register_atom
from sciona.catboost_model_state import pack_catboost_model,unpack_catboost_model
from sciona.catboost_bank_state import pack_model_bank,unpack_model_bank
from sciona.atoms.ml.model_selection.named_regression import witness_materialized


@register_atom(witness=witness_materialized)
def pack_model(model: object) -> dict:
    """Encode native regression state, ordered schema and exact backend version."""
    return pack_catboost_model(model)


@register_atom(witness=witness_materialized)
def unpack_model(state: dict) -> object:
    """Restore verified native state and reject envelope/schema mismatch."""
    return unpack_catboost_model(state)


@register_atom(witness=witness_materialized)
def pack_bank(bank: dict) -> dict:
    """Encode each distinct model once, preserving shared identities and typed slots."""
    return pack_model_bank(bank)


@register_atom(witness=witness_materialized)
def unpack_bank(state: dict) -> dict:
    """Verify all model-bank references before restoring native objects and aliases."""
    return unpack_model_bank(state)
