from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sample_parameter_distributions,
)

@register_atom(witness_sample_parameter_distributions, name="sample_parameter_distributions")
@icontract.require(lambda distributions, n_iter, random_state: n_iter > 0, "Precondition failed: n_iter > 0")
@icontract.ensure(lambda result, distributions, n_iter, random_state: len(sampled_configurations) == n_iter, "Postcondition failed: len(sampled_configurations) == n_iter")
def sample_parameter_distributions(distributions: str, n_iter: int, random_state: int = None) -> str:
    """Draw deterministic random hyperparameter samples from target distributions.

    Args:
        distributions: dict[str, Any]
        n_iter: int
        random_state: Optional[int]

    Returns:
        sampled_configurations: list[dict[str, Any]]
    """
    import sklearn.model_selection
    return sklearn.model_selection.ParameterSampler(distributions=distributions, n_iter=n_iter, random_state=random_state) # type: ignore

