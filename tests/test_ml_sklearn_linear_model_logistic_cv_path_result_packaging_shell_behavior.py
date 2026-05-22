from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_logistic_cv_path_result_packaging_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_path_result_packaging_shell import (
        logistic_cv_coefs_paths_by_class,
        logistic_cv_coefs_paths_layout,
        logistic_cv_n_iter_layout,
        logistic_cv_path_results,
        logistic_cv_public_Cs,
        logistic_cv_scores_by_class,
        logistic_cv_scores_layout,
    )

    assert callable(logistic_cv_path_results)
    assert callable(logistic_cv_public_Cs)
    assert callable(logistic_cv_coefs_paths_layout)
    assert callable(logistic_cv_n_iter_layout)
    assert callable(logistic_cv_scores_layout)
    assert callable(logistic_cv_scores_by_class)
    assert callable(logistic_cv_coefs_paths_by_class)


def test_logistic_cv_path_results_and_public_Cs_match_source_unpacking() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_path_result_packaging_shell import (
        logistic_cv_path_results,
        logistic_cv_public_Cs,
    )

    Cs = np.array([0.1, 1.0], dtype=np.float64)
    path_results = (
        (np.array([[1.0, 2.0]]), Cs, np.array([0.2, 0.3]), np.array([5, 6], dtype=np.int32)),
        (np.array([[3.0, 4.0]]), Cs, np.array([0.4, 0.5]), np.array([7, 8], dtype=np.int32)),
    )

    coefs_paths, Cs_rows, scores, n_iter = logistic_cv_path_results(path_results)
    public_Cs = logistic_cv_public_Cs(Cs_rows)

    assert coefs_paths == (path_results[0][0], path_results[1][0])
    assert Cs_rows == (Cs, Cs)
    assert scores == (path_results[0][2], path_results[1][2])
    assert n_iter == (path_results[0][3], path_results[1][3])
    assert public_Cs is Cs
    np.testing.assert_array_equal(public_Cs, Cs)


def test_logistic_cv_ovr_layouts_and_dict_packaging() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_path_result_packaging_shell import (
        logistic_cv_coefs_paths_by_class,
        logistic_cv_coefs_paths_layout,
        logistic_cv_n_iter_layout,
        logistic_cv_scores_by_class,
        logistic_cv_scores_layout,
    )

    n_classes = 2
    n_folds = 3
    n_Cs = 2
    n_l1_ratios = 1
    width = 4
    classes = ("a", "b")
    coefs_raw = np.arange(n_classes * n_folds * n_Cs * n_l1_ratios * width, dtype=np.float32).reshape(-1, n_Cs, width)
    scores_raw = np.linspace(0.1, 0.9, n_classes * n_folds * n_Cs, dtype=np.float64).reshape(-1, n_Cs)
    n_iter_raw = np.arange(n_classes * n_folds * n_Cs, dtype=np.int32).reshape(-1, n_Cs)

    coefs = logistic_cv_coefs_paths_layout(
        coefs_raw,
        multi_class="ovr",
        n_classes=n_classes,
        n_folds=n_folds,
        n_Cs=n_Cs,
        n_l1_ratios=n_l1_ratios,
    )
    scores = logistic_cv_scores_layout(scores_raw, multi_class="ovr", n_classes=n_classes, n_folds=n_folds)
    n_iter = logistic_cv_n_iter_layout(
        n_iter_raw,
        multi_class="ovr",
        n_classes=n_classes,
        n_folds=n_folds,
        n_Cs=n_Cs,
        n_l1_ratios=n_l1_ratios,
    )

    np.testing.assert_array_equal(coefs, np.reshape(coefs_raw, (n_classes, n_folds, n_Cs, width)))
    assert coefs.dtype == np.float32
    np.testing.assert_array_equal(scores, np.reshape(scores_raw, (n_classes, n_folds, n_Cs)))
    assert scores.dtype == np.float64
    np.testing.assert_array_equal(n_iter, np.reshape(n_iter_raw, (n_classes, n_folds, n_Cs)))
    assert n_iter.dtype == np.int32
    scores_dict = logistic_cv_scores_by_class(classes, scores)
    coefs_dict = logistic_cv_coefs_paths_by_class(classes, coefs)
    assert list(scores_dict) == list(classes)
    np.testing.assert_array_equal(scores_dict["a"], scores[0])
    np.testing.assert_array_equal(coefs_dict["b"], coefs[1])


def test_logistic_cv_multinomial_layouts_tile_scores_across_classes() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_path_result_packaging_shell import (
        logistic_cv_coefs_paths_layout,
        logistic_cv_n_iter_layout,
        logistic_cv_scores_layout,
    )

    n_classes = 3
    n_folds = 2
    n_Cs = 2
    n_l1_ratios = 1
    width = 4
    coefs_raw = np.arange(n_folds * n_Cs * n_l1_ratios * n_classes * width, dtype=np.float64).reshape(-1, n_Cs, n_classes, width)
    scores_raw = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    n_iter_raw = np.array([[5, 6], [7, 8]], dtype=np.int32)

    coefs = logistic_cv_coefs_paths_layout(
        coefs_raw,
        multi_class="multinomial",
        n_classes=n_classes,
        n_folds=n_folds,
        n_Cs=n_Cs,
        n_l1_ratios=n_l1_ratios,
    )
    scores = logistic_cv_scores_layout(scores_raw, multi_class="multinomial", n_classes=n_classes, n_folds=n_folds)
    n_iter = logistic_cv_n_iter_layout(
        n_iter_raw,
        multi_class="multinomial",
        n_classes=n_classes,
        n_folds=n_folds,
        n_Cs=n_Cs,
        n_l1_ratios=n_l1_ratios,
    )

    expected_coefs = np.reshape(coefs_raw, (n_folds, n_Cs, n_classes, width))
    expected_coefs = np.swapaxes(expected_coefs, 0, 1)
    expected_coefs = np.swapaxes(expected_coefs, 0, 2)
    np.testing.assert_array_equal(coefs, expected_coefs)
    np.testing.assert_array_equal(scores, np.reshape(np.tile(scores_raw, (n_classes, 1, 1)), (n_classes, n_folds, n_Cs)))
    np.testing.assert_array_equal(n_iter, np.reshape(n_iter_raw, (1, n_folds, n_Cs)))


def test_logistic_cv_flattened_l1_ratio_axis_is_preserved_before_public_expansion() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_path_result_packaging_shell import (
        logistic_cv_coefs_paths_layout,
        logistic_cv_n_iter_layout,
        logistic_cv_scores_layout,
    )

    n_classes = 1
    n_folds = 2
    n_Cs = 2
    n_l1_ratios = 3
    width = 3
    flattened = n_Cs * n_l1_ratios
    coefs_raw = np.arange(n_classes * n_folds * flattened * width, dtype=np.float64).reshape(-1, flattened, width)
    scores_raw = np.linspace(0.1, 0.9, n_classes * n_folds * flattened, dtype=np.float64).reshape(-1, flattened)
    n_iter_raw = np.arange(n_classes * n_folds * flattened, dtype=np.int32).reshape(-1, flattened)

    coefs = logistic_cv_coefs_paths_layout(
        coefs_raw,
        multi_class="ovr",
        n_classes=n_classes,
        n_folds=n_folds,
        n_Cs=n_Cs,
        n_l1_ratios=n_l1_ratios,
    )
    scores = logistic_cv_scores_layout(scores_raw, multi_class="ovr", n_classes=n_classes, n_folds=n_folds)
    n_iter = logistic_cv_n_iter_layout(
        n_iter_raw,
        multi_class="ovr",
        n_classes=n_classes,
        n_folds=n_folds,
        n_Cs=n_Cs,
        n_l1_ratios=n_l1_ratios,
    )

    assert coefs.shape == (n_classes, n_folds, flattened, width)
    assert scores.shape == (n_classes, n_folds, flattened)
    assert n_iter.shape == (n_classes, n_folds, flattened)


def test_logistic_cv_path_result_packaging_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_path_result_packaging_shell import (
        logistic_cv_coefs_paths_by_class,
        logistic_cv_coefs_paths_layout,
        logistic_cv_n_iter_layout,
        logistic_cv_path_results,
        logistic_cv_public_Cs,
        logistic_cv_scores_by_class,
        logistic_cv_scores_layout,
    )

    with pytest.raises(ViolationError):
        logistic_cv_path_results(())

    with pytest.raises(ViolationError):
        logistic_cv_path_results(((np.array([1.0]), np.array([1.0]), np.array([1.0])),))

    with pytest.raises(ViolationError):
        logistic_cv_public_Cs(())

    with pytest.raises(ViolationError):
        logistic_cv_coefs_paths_layout(
            np.ones((2, 2), dtype=np.float64),
            multi_class="bad",
            n_classes=1,
            n_folds=1,
            n_Cs=2,
            n_l1_ratios=1,
        )

    with pytest.raises(ViolationError):
        logistic_cv_coefs_paths_layout(
            np.ones((5,), dtype=np.float64),
            multi_class="ovr",
            n_classes=2,
            n_folds=2,
            n_Cs=2,
            n_l1_ratios=1,
        )

    with pytest.raises(ViolationError):
        logistic_cv_n_iter_layout(
            np.array([-1], dtype=np.int32),
            multi_class="ovr",
            n_classes=1,
            n_folds=1,
            n_Cs=1,
            n_l1_ratios=1,
        )

    with pytest.raises(ViolationError):
        logistic_cv_scores_layout(np.array([np.nan]), multi_class="ovr", n_classes=1, n_folds=1)

    with pytest.raises(ViolationError):
        logistic_cv_scores_by_class(("a", "b"), np.ones((1, 2), dtype=np.float64))

    with pytest.raises(ViolationError):
        logistic_cv_coefs_paths_by_class(("a", "b"), np.ones((1, 2, 3), dtype=np.float64))
