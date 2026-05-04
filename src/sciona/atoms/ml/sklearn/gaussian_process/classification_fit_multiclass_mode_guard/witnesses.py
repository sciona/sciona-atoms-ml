"""Ghost witnesses for Gaussian-process classification fit multiclass-mode guard atoms."""

from __future__ import annotations


def witness_gpc_fit_require_supported_multiclass_mode(
    n_classes: int,
    multi_class: str,
) -> str:
    """Describe GaussianProcessClassifier.fit's validated multiclass mode."""
    del n_classes
    return multi_class
