from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy import stats


def test_tabular_gradient_boosting_atoms_import() -> None:
    from sciona.atoms.ml.tabular.gradient_boosting import (
        aggregate_child_table,
        extract_pseudo_labels,
        frequency_encode,
        frequency_encode_fit,
        group_aggregate,
        log_cosh_gradient,
        missing_indicator_and_impute,
        null_importance_p_values,
        pairwise_products,
        pairwise_ratios,
        rank_transform,
        rolling_statistics,
        target_encode,
        temporal_difference,
        time_decay_aggregate,
        tweedie_gradient,
    )

    for atom in (
        aggregate_child_table,
        extract_pseudo_labels,
        frequency_encode,
        frequency_encode_fit,
        group_aggregate,
        log_cosh_gradient,
        missing_indicator_and_impute,
        null_importance_p_values,
        pairwise_products,
        pairwise_ratios,
        rank_transform,
        rolling_statistics,
        target_encode,
        temporal_difference,
        time_decay_aggregate,
        tweedie_gradient,
    ):
        assert callable(atom)


def test_group_and_child_aggregations_are_parent_aligned() -> None:
    from sciona.atoms.ml.tabular.gradient_boosting import aggregate_child_table, group_aggregate

    values = np.array([1.0, 3.0, 2.0, 10.0], dtype=np.float64)
    groups = np.array(["b", "a", "b", "a"], dtype=object)

    assert np.allclose(group_aggregate(values, groups, "mean"), np.array([6.5, 1.5]))
    assert np.allclose(group_aggregate(values, groups, "count"), np.array([2.0, 2.0]))

    parent_keys = np.array(["a", "b", "c"], dtype=object)
    child = aggregate_child_table(values, parent_keys, groups, ["mean", "count"])
    assert np.allclose(child[:2], np.array([[6.5, 2.0], [1.5, 2.0]]))
    assert np.isnan(child[2, 0])
    assert child[2, 1] == 0.0


def test_temporal_decay_and_rolling_features() -> None:
    from sciona.atoms.ml.tabular.gradient_boosting import (
        rolling_statistics,
        temporal_difference,
        time_decay_aggregate,
    )

    values = np.array([10.0, 5.0, 12.0, 9.0], dtype=np.float64)
    entity_ids = np.array(["u1", "u2", "u1", "u2"], dtype=object)
    sort_keys = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float64)
    differences = temporal_difference(values, entity_ids, sort_keys)
    assert np.isnan(differences[0])
    assert np.isnan(differences[1])
    assert differences[2] == 2.0
    assert differences[3] == 4.0

    rolled = rolling_statistics(np.array([1.0, 2.0, 4.0, 8.0]), 2, ["mean", "max"])
    assert np.allclose(rolled, np.array([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [6.0, 8.0]]))

    decayed = time_decay_aggregate(
        np.array([1.0, 1.0, 2.0], dtype=np.float64),
        np.array([0.0, 1.0, 1.0], dtype=np.float64),
        np.array(["a", "a", "b"], dtype=object),
        decay_rate=1.0,
    )
    assert np.allclose(decayed, np.array([np.exp(-1.0) + 1.0, 2.0]))


def test_feature_transforms_and_encoders() -> None:
    from sciona.atoms.ml.tabular.gradient_boosting import (
        frequency_encode,
        frequency_encode_fit,
        missing_indicator_and_impute,
        pairwise_products,
        pairwise_ratios,
        rank_transform,
        target_encode,
    )

    X = np.array([[2.0, 4.0, 8.0], [3.0, 6.0, 9.0]], dtype=np.float64)
    assert np.allclose(pairwise_products(X), np.array([[8.0, 16.0, 32.0], [18.0, 27.0, 54.0]]))
    assert np.allclose(pairwise_ratios(X), np.array([[0.5, 0.25, 0.5], [0.5, 1.0 / 3.0, 2.0 / 3.0]]))

    imputed, indicators = missing_indicator_and_impute(np.array([[1.0, np.nan], [np.nan, 4.0]]), fill_value=-1.0)
    assert np.array_equal(indicators, np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert np.array_equal(imputed, np.array([[1.0, -1.0], [-1.0, 4.0]]))

    values = np.array([10.0, 20.0, 20.0, 30.0], dtype=np.float64)
    assert np.allclose(rank_transform(values, "average"), stats.rankdata(values, method="average"))

    categories = np.array(["x", "y", "x", "z"], dtype=object)
    freq_map = frequency_encode_fit(categories)
    assert freq_map == {"x": 0.5, "y": 0.25, "z": 0.25}
    assert np.allclose(frequency_encode(np.array(["x", "z", "q"], dtype=object), freq_map), np.array([0.5, 0.25, 0.0]))

    encodings = target_encode(categories, np.array([1.0, 0.0, 0.0, 1.0]), smoothing=1.0, prior=0.5)
    assert encodings == {"x": 0.5, "y": 0.25, "z": 0.75}


def test_loss_and_selection_helpers() -> None:
    from sciona.atoms.ml.tabular.gradient_boosting import (
        extract_pseudo_labels,
        log_cosh_gradient,
        null_importance_p_values,
        tweedie_gradient,
    )

    gradient, hessian = tweedie_gradient(
        np.array([0.0, 0.5], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float64),
        power=1.5,
    )
    expected_gradient = -np.array([1.0, 2.0]) * np.exp(-0.5 * np.array([0.0, 0.5])) + np.exp(0.5 * np.array([0.0, 0.5]))
    assert np.allclose(gradient, expected_gradient)
    assert np.all(hessian > 0.0)

    log_grad, log_hess = log_cosh_gradient(np.array([1.0, 2.0]), np.array([1.0, 4.0]))
    assert np.allclose(log_grad, np.tanh(np.array([0.0, 2.0])))
    assert np.all(log_hess >= 0.0)

    p_values = null_importance_p_values(
        np.array([0.8, 0.2], dtype=np.float64),
        np.array([[0.1, 0.3], [0.7, 0.2], [0.9, 0.1]], dtype=np.float64),
    )
    assert np.allclose(p_values, np.array([1.0 / 3.0, 2.0 / 3.0]))

    positive, negative = extract_pseudo_labels(np.array([0.01, 0.5, 0.99]), upper_threshold=0.95, lower_threshold=0.05)
    assert np.array_equal(positive, np.array([2]))
    assert np.array_equal(negative, np.array([0]))


def test_contracts_reject_invalid_tabular_inputs() -> None:
    from sciona.atoms.ml.tabular.gradient_boosting import group_aggregate, pairwise_ratios, tweedie_gradient

    with pytest.raises(ViolationError):
        group_aggregate(np.array([1.0, 2.0]), np.array(["a"], dtype=object), "mean")

    with pytest.raises(ViolationError):
        pairwise_ratios(np.array([[1.0, 2.0]]), epsilon=0.0)

    with pytest.raises(ViolationError):
        tweedie_gradient(np.array([0.0]), np.array([1.0]), power=2.0)

