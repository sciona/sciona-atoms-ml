"""One-vs-rest fit bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_one_vs_rest_binary_fit_labels,
    witness_one_vs_rest_class_count,
    witness_one_vs_rest_multilabel_flag,
    witness_one_vs_rest_partial_fit_first_call,
)


def _class_label_valid(value: object) -> bool:
    return isinstance(value, (str, int, float, np.integer, np.floating)) and not (
        isinstance(value, float) and not np.isfinite(value)
    )


def _class_tuple_valid(values: object) -> bool:
    if not isinstance(values, tuple) or len(values) < 1:
        return False
    if not all(_class_label_valid(value) for value in values):
        return False
    normalized = [str(value) if isinstance(value, str) else float(value) if isinstance(value, (float, np.floating)) else int(value) for value in values]
    return len({repr(value) for value in normalized}) == len(normalized)


def _binary_label_pair_valid(result: object, class_label: object) -> bool:
    values = np.asarray(result, dtype=object)
    return bool(values.shape == (2,) and values[0] == f"not {class_label}" and values[1] == class_label)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _bool_result(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_one_vs_rest_binary_fit_labels)
@icontract.require(lambda class_label: _class_label_valid(class_label), "class_label must be a finite numeric or string label")
@icontract.ensure(lambda result, class_label: _binary_label_pair_valid(result, class_label), "binary fit labels must match sklearn's ['not class', class] pair")
def one_vs_rest_binary_fit_labels(class_label: object) -> NDArray[np.object_]:
    """Return sklearn's two-label classes vector for one binary one-vs-rest fit."""
    return np.asarray([f"not {class_label}", class_label], dtype=object)


@register_atom(witness_one_vs_rest_class_count)
@icontract.require(lambda classes: _class_tuple_valid(classes), "classes must be a nonempty tuple of unique finite numeric or string labels")
@icontract.ensure(lambda result: _positive_int(result), "class count must be a positive integer")
def one_vs_rest_class_count(classes: tuple[object, ...]) -> int:
    """Return the fitted one-vs-rest class count from a class-label tuple."""
    return len(classes)


@register_atom(witness_one_vs_rest_multilabel_flag)
@icontract.require(lambda y_type: isinstance(y_type, str) and y_type != "", "y_type must be a nonempty string")
@icontract.ensure(lambda result: _bool_result(result), "multilabel flag must be boolean")
def one_vs_rest_multilabel_flag(y_type: str) -> bool:
    """Return whether a fitted one-vs-rest label binarizer type is multilabel."""
    return y_type.startswith("multilabel")


@register_atom(witness_one_vs_rest_partial_fit_first_call)
@icontract.require(lambda has_estimators: isinstance(has_estimators, bool), "has_estimators must be boolean")
@icontract.ensure(lambda result: _bool_result(result), "first-call flag must be boolean")
def one_vs_rest_partial_fit_first_call(
    has_estimators: bool,
) -> bool:
    """Return whether one-vs-rest partial_fit is entering before estimators are initialized."""
    return not has_estimators
