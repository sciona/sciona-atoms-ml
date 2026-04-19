from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import calibration_curve as sklearn_calibration_curve


def test_calibration_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.calibration import calibration_curve

    assert callable(calibration_curve)


def test_calibration_curve_matches_sklearn_uniform() -> None:
    from sciona.atoms.ml.sklearn.calibration import calibration_curve

    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
    result = calibration_curve(y_true, y_prob, n_bins=3)
    expected = sklearn_calibration_curve(y_true, y_prob, n_bins=3)
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])


def test_calibration_curve_matches_sklearn_quantile() -> None:
    from sciona.atoms.ml.sklearn.calibration import calibration_curve

    y_true = np.array([False, False, True, False, True, True, True, False])
    y_prob = np.array([0.01, 0.15, 0.2, 0.35, 0.55, 0.7, 0.8, 0.95])
    result = calibration_curve(y_true, y_prob, n_bins=4, strategy="quantile")
    expected = sklearn_calibration_curve(y_true, y_prob, n_bins=4, strategy="quantile")
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])


def test_calibration_curve_matches_sklearn_string_pos_label() -> None:
    from sciona.atoms.ml.sklearn.calibration import calibration_curve

    y_true = np.array(["no", "yes", "no", "yes", "yes", "no"])
    y_prob = np.array([0.05, 0.2, 0.4, 0.8, 0.9, 0.6])
    result = calibration_curve(y_true, y_prob, pos_label="yes", n_bins=3)
    expected = sklearn_calibration_curve(y_true, y_prob, pos_label="yes", n_bins=3)
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])


def test_calibration_curve_omits_empty_bins_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.calibration import calibration_curve

    y_true = np.array([0, 1, 1])
    y_prob = np.array([0.05, 0.9, 0.95])
    result = calibration_curve(y_true, y_prob, n_bins=10)
    expected = sklearn_calibration_curve(y_true, y_prob, n_bins=10)
    assert result[0].shape[0] < 10
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])


def test_calibration_curve_multiclass_error_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.calibration import calibration_curve

    y_true = np.array([0, 1, 2])
    y_prob = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="Only binary classification"):
        calibration_curve(y_true, y_prob, pos_label=1)
    with pytest.raises(ValueError, match="Only binary classification"):
        sklearn_calibration_curve(y_true, y_prob, pos_label=1)


def test_calibration_curve_rejects_bad_probability_range() -> None:
    from sciona.atoms.ml.sklearn.calibration import calibration_curve

    with pytest.raises(Exception):
        calibration_curve(np.array([0, 1]), np.array([0.2, 1.2]))
