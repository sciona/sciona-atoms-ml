"""Gaussian-process classification fit multiclass-mode guard atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import witness_gpc_fit_require_supported_multiclass_mode


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value != "")


@register_atom(witness_gpc_fit_require_supported_multiclass_mode)
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
@icontract.require(lambda multi_class: _nonempty_string(multi_class), "multi_class must be a nonempty string")
@icontract.ensure(lambda result, multi_class: result == multi_class, "validated multiclass mode must preserve the supplied mode string")
def gpc_fit_require_supported_multiclass_mode(
    n_classes: int,
    multi_class: str,
) -> str:
    """Apply GaussianProcessClassifier.fit's unknown multi-class mode guard."""
    if int(n_classes) > 2 and multi_class not in {"one_vs_rest", "one_vs_one"}:
        raise ValueError("Unknown multi-class mode %s" % multi_class)
    return multi_class
