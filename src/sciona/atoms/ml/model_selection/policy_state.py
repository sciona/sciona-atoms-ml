"""Reusable integrity binding for model state and caller execution policies."""
from sciona.ghost.registry import register_atom
from sciona.policy_bound_state import bind_policy_state,unbind_policy_state
from sciona.atoms.ml.model_selection.named_regression import witness_materialized


@register_atom(witness=witness_materialized)
def bind(model_state: dict, policy: dict) -> dict:
    """Own and bind JSON model state with its complete execution policy."""
    return bind_policy_state(model_state,policy)


@register_atom(witness=witness_materialized)
def unbind(state: dict) -> tuple:
    """Reject binding corruption before returning owned model state and policy."""
    return unbind_policy_state(state)
