from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.preprocessing import LabelBinarizer


def test_one_vs_rest_targets_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_targets import (
        one_vs_rest_fit_classes,
        one_vs_rest_fit_target_indicator_csc,
        one_vs_rest_partial_fit_target_indicator_csc,
        one_vs_rest_partial_fit_unknown_classes,
        one_vs_rest_target_columns_dense,
    )

    assert callable(one_vs_rest_fit_classes)
    assert callable(one_vs_rest_fit_target_indicator_csc)
    assert callable(one_vs_rest_partial_fit_unknown_classes)
    assert callable(one_vs_rest_partial_fit_target_indicator_csc)
    assert callable(one_vs_rest_target_columns_dense)


def test_fit_classes_match_label_binarizer_for_multiclass_and_binary() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_targets import one_vs_rest_fit_classes

    multiclass_y = np.array([2.0, 1.0, 2.0, 0.0], dtype=np.float64)
    binary_y = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float64)

    assert np.array_equal(one_vs_rest_fit_classes(multiclass_y), np.array([0.0, 1.0, 2.0], dtype=np.float64))
    assert np.array_equal(one_vs_rest_fit_classes(binary_y), np.array([0.0, 1.0], dtype=np.float64))


def test_fit_target_indicator_matches_sparse_label_binarizer() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_targets import one_vs_rest_fit_target_indicator_csc

    y = np.array([2.0, 0.0, 2.0, 1.0], dtype=np.float64)
    expected = LabelBinarizer(sparse_output=True).fit_transform(y).tocsc()

    observed = one_vs_rest_fit_target_indicator_csc(y)

    assert np.array_equal(observed.toarray(), expected.toarray())


def test_partial_fit_unknown_classes_returns_sorted_unique_labels() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_targets import one_vs_rest_partial_fit_unknown_classes

    y = np.array([2.0, 5.0, 5.0, 4.0, 0.0], dtype=np.float64)
    classes = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    observed = one_vs_rest_partial_fit_unknown_classes(y, classes)

    assert np.array_equal(observed, np.array([4.0, 5.0], dtype=np.float64))


def test_partial_fit_target_indicator_matches_label_binarizer_transform() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_targets import one_vs_rest_partial_fit_target_indicator_csc

    classes = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([2.0, 0.0, 2.0, 1.0], dtype=np.float64)
    lb = LabelBinarizer(sparse_output=True)
    lb.fit(classes)
    expected = lb.transform(y).tocsc()

    observed = one_vs_rest_partial_fit_target_indicator_csc(y, classes)

    assert np.array_equal(observed.toarray(), expected.toarray())


def test_target_columns_dense_matches_sklearn_column_materialization() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_targets import (
        one_vs_rest_fit_target_indicator_csc,
        one_vs_rest_target_columns_dense,
    )

    y = np.array([2.0, 0.0, 2.0, 1.0], dtype=np.float64)
    indicator = one_vs_rest_fit_target_indicator_csc(y)

    observed = one_vs_rest_target_columns_dense(indicator)
    expected = np.asarray([col.toarray().ravel() for col in indicator.T], dtype=np.float64)

    assert np.array_equal(observed, expected)


def test_partial_fit_target_indicator_contract_rejects_unknown_classes() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_targets import one_vs_rest_partial_fit_target_indicator_csc

    with pytest.raises(ViolationError):
        one_vs_rest_partial_fit_target_indicator_csc(
            np.array([0.0, 3.0], dtype=np.float64),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
        )
