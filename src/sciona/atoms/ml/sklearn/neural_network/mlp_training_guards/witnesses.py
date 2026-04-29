"""Ghost witnesses for MLP training-guard helper atoms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sciona.ghost.abstract import AbstractArray


def witness_mlp_partial_fit_require_stochastic_solver(
    *,
    solver: str,
) -> bool:
    """Describe the partial_fit solver-eligibility guard."""
    del solver
    return True


def witness_mlp_fit_require_finite_weights(
    coefs: Sequence[AbstractArray],
    intercepts: Sequence[AbstractArray],
) -> bool:
    """Describe the post-fit finite-weights guard after an MLP solver run."""
    del coefs
    del intercepts
    return bool(np.bool_(True))
