import numpy as np
import pytest
from scipy.stats import ks_2samp
from sciona.atoms.ml.group_distribution_matching import group_subset_ks, witness_group_subset_ks
from sciona.ghost.abstract import AbstractArray


def inputs():
    # Unequal group sizes deliberately distinguish row weighting from group weighting.
    return (np.array([[0., 0.], [0., 2.], [0., 4.], [10., 6.]]),
            np.array([[0., 0.], [10., 6.]]), np.array([7, 7, 7, 9]),
            np.array([[True, False], [False, True], [True, True], [True, False]]))


def test_independent_scipy_oracle_group_weighting_and_duplicate_order():
    X, R, groups, masks = inputs()
    copies = [v.copy() for v in (X, R, groups, masks)]
    actual = group_subset_ks(X, R, groups, masks, [7, 9], ['a', 'b'], 'u')
    expected = np.array([[ks_2samp(X[np.isin(groups, np.array([7, 9])[mask]), j], R[:, j]).statistic
                          for j in range(2)] for mask in masks])
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    assert actual[2, 0] == .25  # Equal group weighting would incorrectly produce zero.
    np.testing.assert_array_equal(actual[0], actual[3])
    for original, copy in zip((X, R, groups, masks), copies):
        np.testing.assert_array_equal(original, copy)


def test_group_and_row_permutation_and_monotone_unit_conversion():
    X, R, groups, masks = inputs()
    base = group_subset_ks(X, R, groups, masks, [7, 9], ['a', 'b'], 'u')
    permutation = [3, 1, 0, 2]
    actual = group_subset_ks(X[permutation], R[::-1], groups[permutation], masks[:, ::-1], [9, 7], ['a', 'b'], 'u')
    np.testing.assert_allclose(actual, base, atol=1e-14, rtol=0)
    scaled = group_subset_ks(X*8+3, R*8+3, groups, masks, [7, 9], ['a', 'b'], 'rescaled')
    np.testing.assert_array_equal(scaled, base)


@pytest.mark.parametrize('change', ['missing_group', 'extra_group', 'empty_subset', 'float_group', 'nan', 'width', 'nonboolean'])
def test_invalid_selection_contracts(change):
    X, R, groups, masks = inputs()
    labels = [7, 9]
    if change == 'missing_group': labels = [7, 8]
    if change == 'extra_group': labels = [7, 9, 10]
    if change == 'empty_subset': masks[0] = False
    if change == 'float_group': groups = groups.astype(float)
    if change == 'nan': X[0, 0] = np.nan
    if change == 'width': R = R[:, :1]
    if change == 'nonboolean': masks = masks.astype(int)
    with pytest.raises(ValueError):
        group_subset_ks(X, R, groups, masks, labels, ['a', 'b'], 'u')


def test_witness_shape_and_distance_bounds():
    arrays = [AbstractArray(shape=x.shape, dtype=str(x.dtype)) for x in inputs()]
    out = witness_group_subset_ks(*arrays, [7, 9], ['a', 'b'], 'u')
    assert out.shape == (4, 2) and out.min_val == 0 and out.max_val == 1
