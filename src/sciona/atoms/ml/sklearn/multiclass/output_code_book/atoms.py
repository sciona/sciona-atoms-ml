"""Output-code book construction helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_output_code_discrete_book,
    witness_output_code_uniform_book,
)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _seed_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool))


def _uniform_book_valid(uniform_book: object) -> bool:
    try:
        array = np.asarray(uniform_book, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.all(np.isfinite(array))
        and np.all(array >= 0.0)
        and np.all(array < 1.0)
    )


@register_atom(witness_output_code_uniform_book)
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
@icontract.require(lambda n_estimators: _positive_int(n_estimators), "n_estimators must be a positive integer")
@icontract.require(lambda seed: _seed_int(seed), "seed must be an integer")
@icontract.ensure(lambda result, n_classes, n_estimators: result.shape == (n_classes, n_estimators), "result must match the requested code-book shape")
@icontract.ensure(lambda result: _uniform_book_valid(result), "result must be finite uniform draws in [0, 1)")
def output_code_uniform_book(
    *,
    n_classes: int,
    n_estimators: int,
    seed: int,
) -> NDArray[np.float64]:
    """Generate sklearn's seeded uniform output-code book before discretization."""
    random_state = np.random.RandomState(seed)
    return np.asarray(
        random_state.uniform(size=(n_classes, n_estimators)),
        dtype=np.float64,
    )


@register_atom(witness_output_code_discrete_book)
@icontract.require(lambda uniform_book: _uniform_book_valid(uniform_book), "uniform_book must be a finite matrix of draws in [0, 1)")
@icontract.require(lambda has_decision_function: isinstance(has_decision_function, bool), "has_decision_function must be boolean")
@icontract.ensure(lambda result, uniform_book: result.shape == np.asarray(uniform_book).shape, "result must preserve code-book shape")
@icontract.ensure(
    lambda result, has_decision_function: set(np.unique(result)).issubset({-1.0, 0.0, 1.0} if has_decision_function else {0.0, 1.0}),
    "result must use sklearn's discrete code values",
)
def output_code_discrete_book(
    uniform_book: NDArray[np.float64],
    *,
    has_decision_function: bool,
) -> NDArray[np.float64]:
    """Discretize a uniform output-code book with sklearn's threshold and sign rules."""
    code_book = np.asarray(uniform_book, dtype=np.float64).copy()
    code_book[code_book > 0.5] = 1.0
    if has_decision_function:
        code_book[code_book != 1.0] = -1.0
    else:
        code_book[code_book != 1.0] = 0.0
    return code_book
