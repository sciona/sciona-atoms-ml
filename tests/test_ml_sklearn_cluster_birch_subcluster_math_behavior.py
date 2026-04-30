from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster._birch import _CFSubcluster

from sciona.atoms.ml.sklearn.cluster.birch_subcluster_math import (
    BirchSubclusterStats,
    birch_subcluster_merge,
    birch_subcluster_radius,
    birch_subcluster_singleton,
    birch_subcluster_squared_radius,
    birch_subcluster_update,
)


def _stats_from_sklearn(subcluster: _CFSubcluster) -> BirchSubclusterStats:
    return BirchSubclusterStats(
        n_samples=int(subcluster.n_samples_),
        linear_sum=np.asarray(subcluster.linear_sum_, dtype=np.float64).copy(),
        squared_sum=float(subcluster.squared_sum_),
        centroid=np.asarray(subcluster.centroid_, dtype=np.float64).copy(),
        sq_norm=float(subcluster.sq_norm_),
    )


def _assert_stats_equal(left: BirchSubclusterStats, right: BirchSubclusterStats) -> None:
    assert left.n_samples == right.n_samples
    assert np.allclose(left.linear_sum, right.linear_sum)
    assert np.isclose(left.squared_sum, right.squared_sum)
    assert np.allclose(left.centroid, right.centroid)
    assert np.isclose(left.sq_norm, right.sq_norm)


def test_birch_subcluster_math_atoms_import() -> None:
    assert callable(birch_subcluster_singleton)
    assert callable(birch_subcluster_update)
    assert callable(birch_subcluster_squared_radius)
    assert callable(birch_subcluster_merge)
    assert callable(birch_subcluster_radius)


def test_birch_subcluster_singleton_matches_sklearn_init() -> None:
    vector = np.array([2.0, -1.0, 3.0], dtype=np.float64)
    expected = _stats_from_sklearn(_CFSubcluster(linear_sum=vector.copy()))
    observed = birch_subcluster_singleton(vector)
    _assert_stats_equal(observed, expected)


def test_birch_subcluster_update_matches_sklearn_update() -> None:
    base = _CFSubcluster(linear_sum=np.array([1.0, 2.0], dtype=np.float64))
    added = _CFSubcluster(linear_sum=np.array([3.0, -1.0], dtype=np.float64))
    base.update(added)

    observed = birch_subcluster_update(
        birch_subcluster_singleton(np.array([1.0, 2.0], dtype=np.float64)),
        birch_subcluster_singleton(np.array([3.0, -1.0], dtype=np.float64)),
    )
    _assert_stats_equal(observed, _stats_from_sklearn(base))


def test_birch_subcluster_squared_radius_and_radius_match_sklearn_property() -> None:
    base = _CFSubcluster(linear_sum=np.array([1.0, 0.0], dtype=np.float64))
    base.update(_CFSubcluster(linear_sum=np.array([3.0, 0.0], dtype=np.float64)))
    state = _stats_from_sklearn(base)

    expected_sq_radius = base.squared_sum_ / base.n_samples_ - base.sq_norm_
    assert np.isclose(birch_subcluster_squared_radius(state), expected_sq_radius)
    assert np.isclose(birch_subcluster_radius(state), base.radius)


def test_birch_subcluster_merge_matches_sklearn_accept_and_reject_paths() -> None:
    accept_base = _CFSubcluster(linear_sum=np.array([0.0, 0.0], dtype=np.float64))
    accept_nominee = _CFSubcluster(linear_sum=np.array([0.1, 0.0], dtype=np.float64))
    expected_accept = accept_base.merge_subcluster(accept_nominee, threshold=0.2)
    observed_accept, observed_accept_state = birch_subcluster_merge(
        birch_subcluster_singleton(np.array([0.0, 0.0], dtype=np.float64)),
        birch_subcluster_singleton(np.array([0.1, 0.0], dtype=np.float64)),
        threshold=0.2,
    )
    assert observed_accept is expected_accept is True
    _assert_stats_equal(observed_accept_state, _stats_from_sklearn(accept_base))

    reject_base = _CFSubcluster(linear_sum=np.array([0.0, 0.0], dtype=np.float64))
    reject_nominee = _CFSubcluster(linear_sum=np.array([2.0, 0.0], dtype=np.float64))
    original_reject_state = _stats_from_sklearn(reject_base)
    expected_reject = reject_base.merge_subcluster(reject_nominee, threshold=0.2)
    observed_reject, observed_reject_state = birch_subcluster_merge(
        birch_subcluster_singleton(np.array([0.0, 0.0], dtype=np.float64)),
        birch_subcluster_singleton(np.array([2.0, 0.0], dtype=np.float64)),
        threshold=0.2,
    )
    assert observed_reject is expected_reject is False
    _assert_stats_equal(observed_reject_state, original_reject_state)


def test_birch_subcluster_math_contracts_reject_invalid_inputs() -> None:
    with pytest.raises(ViolationError):
        birch_subcluster_singleton(np.array([], dtype=np.float64))

    with pytest.raises(ViolationError):
        birch_subcluster_update(
            birch_subcluster_singleton(np.array([1.0], dtype=np.float64)),
            birch_subcluster_singleton(np.array([1.0, 2.0], dtype=np.float64)),
        )

    with pytest.raises(ViolationError):
        birch_subcluster_merge(
            birch_subcluster_singleton(np.array([1.0], dtype=np.float64)),
            birch_subcluster_singleton(np.array([2.0], dtype=np.float64)),
            threshold=0.0,
        )
