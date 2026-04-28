from __future__ import annotations

import numpy as np


def test_partial_dependence_input_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping import (
        partial_dependence_boolean_categorical_mask,
        partial_dependence_default_categorical_mask,
        partial_dependence_feature_name_index,
        partial_dependence_index_categorical_mask,
        partial_dependence_name_categorical_mask,
        partial_dependence_nonnegative_int_features,
    )

    assert callable(partial_dependence_feature_name_index)
    assert callable(partial_dependence_nonnegative_int_features)
    assert callable(partial_dependence_default_categorical_mask)
    assert callable(partial_dependence_boolean_categorical_mask)
    assert callable(partial_dependence_index_categorical_mask)
    assert callable(partial_dependence_name_categorical_mask)


def test_feature_name_index_matches_sklearn_helper() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping import (
        partial_dependence_feature_name_index,
    )

    feature_names = ("sepal_length", "sepal_width", "petal_length")
    assert partial_dependence_feature_name_index("petal_length", feature_names) == 2


def test_nonnegative_int_features_preserves_valid_vector() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping import (
        partial_dependence_nonnegative_int_features,
    )

    features = np.array([0, 2, 4], dtype=np.int64)
    assert np.array_equal(
        partial_dependence_nonnegative_int_features(features, n_features=5),
        features,
    )


def test_default_categorical_mask_matches_source() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping import (
        partial_dependence_default_categorical_mask,
    )

    assert partial_dependence_default_categorical_mask(3) == (False, False, False)


def test_boolean_categorical_mask_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping import (
        partial_dependence_boolean_categorical_mask,
    )

    categorical = np.array([False, True, False, True], dtype=np.bool_)
    features_indices = np.array([3, 0, 1], dtype=np.int64)

    assert partial_dependence_boolean_categorical_mask(
        categorical, features_indices, n_features=4
    ) == (True, False, True)


def test_index_categorical_mask_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping import (
        partial_dependence_index_categorical_mask,
    )

    categorical_indices = np.array([1, 4, 7], dtype=np.int64)
    features_indices = np.array([7, 2, 1, 0], dtype=np.int64)

    assert partial_dependence_index_categorical_mask(
        categorical_indices, features_indices
    ) == (True, False, True, False)


def test_name_categorical_mask_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping import (
        partial_dependence_name_categorical_mask,
    )

    feature_names = ("age", "zip_code", "income", "state")
    categorical_names = ("state", "zip_code")
    features_indices = np.array([2, 3, 1], dtype=np.int64)

    assert partial_dependence_name_categorical_mask(
        categorical_names,
        features_indices,
        feature_names,
    ) == (False, True, True)
