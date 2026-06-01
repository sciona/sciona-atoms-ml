from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_propose_next_parameters,
)

@register_atom(witness_propose_next_parameters, name="propose_next_parameters")
@icontract.require(lambda history, param_specs: history is not None, "Precondition failed: history is not None")
@icontract.ensure(lambda result, history, param_specs: result is not None, "Postcondition failed: result is not None")
def propose_next_parameters(history: Any, param_specs: str) -> str:
    """Query surrogate model (e.g., TPE Sampler) for the next hyperparameter set to evaluate.

    Args:
        history: list of evaluated trial records
        param_specs: dict[str, Any]

    Returns:
        next_params: dict[str, Any]
    """
    import optuna.samplers.TPESampler
    return optuna.samplers.TPESampler.sample_independent(history=history, param_specs=param_specs) # type: ignore

