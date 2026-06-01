from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_sample_parameter_distributions(distributions: AbstractScalar | str, n_iter: AbstractScalar | int, random_state: AbstractScalar | int) -> AbstractScalar:
    """Ghost witness for sample_parameter_distributions."""
    _ = (distributions, n_iter, random_state)
    return AbstractScalar(dtype="float64")

