"""Training/inference feature-policy binding for portable native state."""
from sciona.ghost.registry import register_atom
from sciona.nasa_first_policy_state import capture_bank_policy,bound_prediction_inputs
from sciona.atoms.ml.model_selection.named_regression import witness_materialized


@register_atom(witness=witness_materialized)
def capture(bank: dict, raw_populations: dict) -> dict:
    """Capture fixed feature, fill, vocabulary and calendar policies matching each model schema."""
    return capture_bank_policy(bank,raw_populations)


@register_atom(witness=witness_materialized)
def prediction_inputs(bank: dict, policy: dict, population: str, raw_options: dict) -> tuple:
    """Use bound policies and reject caller policy drift before prediction fallback routing."""
    return bound_prediction_inputs(bank,policy,population,raw_options)
