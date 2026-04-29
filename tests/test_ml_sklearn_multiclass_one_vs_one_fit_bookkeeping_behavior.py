from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.multiclass.one_vs_one_fit_bookkeeping import (
    one_vs_one_fit_classes,
    one_vs_one_fit_pairwise_indices,
    one_vs_one_fit_require_multiple_classes,
)


def test_one_vs_one_fit_classes_matches_sklearn_unique_order() -> None:
    y = np.array([3.0, 1.0, 3.0, 2.0, 1.0], dtype=np.float64)
    observed = one_vs_one_fit_classes(y)
    assert np.array_equal(observed, np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_one_vs_one_fit_require_multiple_classes_matches_guard() -> None:
    classes = np.array([1.0, 2.0], dtype=np.float64)
    assert np.array_equal(one_vs_one_fit_require_multiple_classes(classes), classes)

    with pytest.raises(ValueError, match="only one class is present"):
        one_vs_one_fit_require_multiple_classes(np.array([1.0], dtype=np.float64))


def test_one_vs_one_fit_pairwise_indices_matches_pairwise_tag_gate() -> None:
    classes = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    pairwise_indices = ((0, 2, 4), (1, 3), (0, 1, 5))

    assert one_vs_one_fit_pairwise_indices(classes, pairwise_indices, pairwise=True) == pairwise_indices
    assert one_vs_one_fit_pairwise_indices(classes, pairwise_indices, pairwise=False) is None


def test_one_vs_one_fit_bookkeeping_rejects_invalid_inputs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        one_vs_one_fit_classes(np.array([[1.0, 2.0]], dtype=np.float64))

    with pytest.raises((ViolationError, ValueError)):
        one_vs_one_fit_pairwise_indices(
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            ((0, 1),),
            pairwise=True,
        )
