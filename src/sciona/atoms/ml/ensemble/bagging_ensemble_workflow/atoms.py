from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_generate_bootstrap_indices,
    witness_fit_bootstrap_estimator,
)

@register_atom(witness_generate_bootstrap_indices, name="generate_bootstrap_indices")
@icontract.require(lambda n_samples, random_state: n_samples > 0, "Precondition failed: n_samples > 0")
@icontract.ensure(lambda result, n_samples, random_state: len(bootstrap_indices) == n_samples, "Postcondition failed: len(bootstrap_indices) == n_samples")
def generate_bootstrap_indices(n_samples: int, random_state: int) -> NDArray[np.int64]:
    """Draw bootstrap samples with replacement and compile OOB exclusions.

    Args:
        n_samples: int
        random_state: int

    Returns:
        bootstrap_indices: NDArray[np.int64]
    """
    import numpy.random
    return numpy.random.choice(n_samples=n_samples, random_state=random_state) # type: ignore

@register_atom(witness_fit_bootstrap_estimator, name="fit_bootstrap_estimator")
@icontract.require(lambda estimator_template, features, targets, bootstrap_indices: estimator_template is not None, "Precondition failed: estimator_template is not None")
@icontract.ensure(lambda result, estimator_template, features, targets, bootstrap_indices: result is not None, "Postcondition failed: result is not None")
def fit_bootstrap_estimator(estimator_template: Any, features: NDArray[np.float64], targets: NDArray[Any], bootstrap_indices: NDArray[np.int64]) -> Any:
    """Train base estimator on bootstrap data.

    Args:
        estimator_template: Any
        features: NDArray[np.float64]
        targets: NDArray[Any]
        bootstrap_indices: NDArray[np.int64]

    Returns:
        fitted_estimator: Any
    """
    import sklearn.base
    return sklearn.base.clone(estimator_template=estimator_template, features=features, targets=targets, bootstrap_indices=bootstrap_indices) # type: ignore

