"""Deterministic MLP training guard helpers."""

from .atoms import (
    mlp_fit_require_finite_weights,
    mlp_partial_fit_require_stochastic_solver,
)

__all__ = [
    "mlp_fit_require_finite_weights",
    "mlp_partial_fit_require_stochastic_solver",
]
