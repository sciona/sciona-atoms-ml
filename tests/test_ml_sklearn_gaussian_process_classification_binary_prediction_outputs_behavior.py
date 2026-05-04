from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.gaussian_process._gpc import _BinaryGaussianProcessClassifierLaplace
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


def _predictive_model() -> tuple[_BinaryGaussianProcessClassifierLaplace, np.ndarray]:
    X = np.array([[-1.0], [-0.3], [0.2], [0.9], [1.4]], dtype=np.float64)
    y = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    kernel = ConstantKernel(1.1, constant_value_bounds="fixed") * RBF(0.9, length_scale_bounds="fixed")
    model = _BinaryGaussianProcessClassifierLaplace(
        kernel=kernel,
        optimizer=None,
        max_iter_predict=25,
        warm_start=False,
    )
    model.fit(X, y)
    X_test = np.array([[-0.7], [0.0], [1.1]], dtype=np.float64)
    return model, X_test


def test_classification_binary_prediction_outputs_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_prediction_outputs import (
        gpc_binary_predict_labels,
        gpc_binary_predict_positive_class_mask,
    )

    assert callable(gpc_binary_predict_positive_class_mask)
    assert callable(gpc_binary_predict_labels)


def test_gpc_binary_predict_positive_class_mask_uses_strict_zero_threshold() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_prediction_outputs import (
        gpc_binary_predict_positive_class_mask,
    )

    f_star = np.array([-0.2, 0.0, 0.4], dtype=np.float64)

    assert np.array_equal(
        gpc_binary_predict_positive_class_mask(f_star),
        np.array([False, False, True], dtype=np.bool_),
    )


def test_gpc_binary_predict_labels_match_private_predict() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_prediction_outputs import (
        gpc_binary_predict_labels,
        gpc_binary_predict_positive_class_mask,
    )

    model, X_test = _predictive_model()
    K_star = model.kernel_(model.X_train_, X_test)
    f_star = K_star.T.dot(model.y_train_ - model.pi_)
    positive_mask = gpc_binary_predict_positive_class_mask(f_star)

    assert np.array_equal(
        gpc_binary_predict_labels(positive_mask, model.classes_),
        model.predict(X_test),
    )


def test_gpc_binary_predict_labels_support_object_class_labels() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_prediction_outputs import (
        gpc_binary_predict_labels,
    )

    positive_mask = np.array([False, True, True, False], dtype=np.bool_)
    classes = np.array(["cold", "hot"], dtype=object)

    assert np.array_equal(
        gpc_binary_predict_labels(positive_mask, classes),
        np.array(["cold", "hot", "hot", "cold"], dtype=object),
    )


def test_gpc_binary_prediction_outputs_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_binary_prediction_outputs import (
        gpc_binary_predict_labels,
        gpc_binary_predict_positive_class_mask,
    )

    with pytest.raises(ViolationError):
        gpc_binary_predict_positive_class_mask(np.array([0.2, np.nan], dtype=np.float64))

    with pytest.raises(ViolationError):
        gpc_binary_predict_labels(np.array([1, 0], dtype=np.int64), np.array([0, 1]))

    with pytest.raises(ViolationError):
        gpc_binary_predict_labels(np.array([True, False], dtype=np.bool_), np.array([0, 1, 2]))
