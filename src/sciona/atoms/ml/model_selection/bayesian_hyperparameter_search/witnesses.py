from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_propose_next_parameters(history: AbstractScalar | Any, param_specs: AbstractScalar | str) -> AbstractScalar:
    """Ghost witness for propose_next_parameters."""
    _ = (history, param_specs)
    return AbstractScalar(dtype="float64")

