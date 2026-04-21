from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_selection import mutual_info_classif as sklearn_mutual_info_classif
from sklearn.feature_selection import mutual_info_regression as sklearn_mutual_info_regression
from sklearn.metrics.cluster import mutual_info_score


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 2.0],
            [3.0, 1.0, 4.0],
            [4.0, 2.0, 4.0],
            [5.0, 2.0, 6.0],
        ],
        dtype=np.float64,
    )
    y_reg = np.array([0.0, 1.0, 1.9, 3.1, 4.2, 5.1], dtype=np.float64)
    y_class = np.array([0, 0, 1, 1, 2, 2])
    return X, y_reg, y_class


def test_mutual_info_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_selection import (
        mutual_info_classif,
        mutual_info_continuous_continuous,
        mutual_info_continuous_discrete,
        mutual_info_pair,
        mutual_info_regression,
    )

    assert callable(mutual_info_continuous_continuous)
    assert callable(mutual_info_continuous_discrete)
    assert callable(mutual_info_pair)
    assert callable(mutual_info_regression)
    assert callable(mutual_info_classif)


def test_pairwise_mutual_information_matches_sklearn_discrete_score() -> None:
    from sciona.atoms.ml.sklearn.feature_selection import mutual_info_pair

    x = np.array([0, 0, 1, 1, 2, 2], dtype=np.float64)
    y = np.array([1, 1, 0, 0, 2, 2], dtype=np.float64)

    assert mutual_info_pair(x, y, x_discrete=True, y_discrete=True) == pytest.approx(mutual_info_score(x, y))


def test_mutual_info_regression_matches_sklearn_continuous_features() -> None:
    from sciona.atoms.ml.sklearn.feature_selection import mutual_info_regression

    X, y_reg, _y_class = _data()
    result = mutual_info_regression(X, y_reg, random_state=0)
    expected = sklearn_mutual_info_regression(X, y_reg, random_state=0)

    assert np.allclose(result, expected)


def test_mutual_info_classif_matches_sklearn_continuous_features() -> None:
    from sciona.atoms.ml.sklearn.feature_selection import mutual_info_classif

    X, _y_reg, y_class = _data()
    result = mutual_info_classif(X, y_class, random_state=0)
    expected = sklearn_mutual_info_classif(X, y_class, random_state=0)

    assert np.allclose(result, expected)


def test_mutual_info_mixed_discrete_features_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    X, y_reg, y_class = _data()

    reg_result = mutual_info_regression(X, y_reg, discrete_features=(1,), random_state=0)
    reg_expected = sklearn_mutual_info_regression(X, y_reg, discrete_features=[1], random_state=0)
    assert np.allclose(reg_result, reg_expected)

    class_result = mutual_info_classif(X, y_class, discrete_features=(False, True, False), random_state=0)
    class_expected = sklearn_mutual_info_classif(X, y_class, discrete_features=[False, True, False], random_state=0)
    assert np.allclose(class_result, class_expected)


def test_mutual_info_rejects_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    X, y_reg, y_class = _data()

    with pytest.raises(Exception):
        mutual_info_regression(X[:, 0], y_reg)
    with pytest.raises(Exception):
        mutual_info_regression(X, y_reg[:-1])
    with pytest.raises(Exception):
        mutual_info_regression(X, y_reg, n_neighbors=0)
    with pytest.raises(Exception):
        mutual_info_regression(X, y_reg, discrete_features=(False, True))
    with pytest.raises(Exception):
        mutual_info_classif(X, y_reg, random_state=0)
    with pytest.raises(Exception):
        mutual_info_classif(X, y_class, discrete_features=(4,))
