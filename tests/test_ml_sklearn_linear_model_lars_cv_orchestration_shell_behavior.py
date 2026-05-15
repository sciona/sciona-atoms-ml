from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import sklearn.linear_model._least_angle as least_angle
from icontract import ViolationError
from sklearn.linear_model import LarsCV, LassoLarsCV
from sklearn.model_selection import KFold


def _make_regression_fixture() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [1.0, 2.0, -0.5, 1.5],
            [0.0, -1.0, 2.0, 0.25],
            [1.5, 0.5, 0.0, -1.0],
            [2.0, 1.0, 1.0, 0.0],
            [-1.0, 1.5, 0.5, 2.0],
            [0.75, -0.5, 1.25, -1.5],
            [1.25, 0.0, -1.0, 0.5],
            [-0.5, 1.0, 2.0, 1.0],
            [2.5, -1.5, 0.25, 0.0],
        ],
        dtype=np.float64,
    )
    y = np.array([2.5, -0.5, 1.75, 2.0, 0.25, -1.0, 0.5, 1.0, 1.25], dtype=np.float64)
    return X, y


def test_lars_cv_orchestration_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv_orchestration_shell import (
        lars_cv_path_residues_callback_kwargs,
    )

    assert callable(lars_cv_path_residues_callback_kwargs)


@pytest.mark.parametrize(
    ("estimator", "method", "precompute", "verbose", "positive"),
    [
        (LarsCV(cv=KFold(n_splits=3), max_iter=5, verbose=3, precompute=True), "lar", True, 3, False),
        (
            LassoLarsCV(cv=KFold(n_splits=3), max_iter=5, verbose=0, precompute=True, positive=True),
            "lasso",
            True,
            0,
            True,
        ),
    ],
)
def test_lars_cv_path_residues_callback_kwargs_match_fit_callback(
    monkeypatch: pytest.MonkeyPatch,
    estimator: LarsCV,
    method: str,
    precompute: object,
    verbose: int,
    positive: bool,
) -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv_orchestration_shell import (
        lars_cv_path_residues_callback_kwargs,
    )

    X, y = _make_regression_fixture()
    captured_kwargs: list[dict[str, object]] = []

    def fake_lars_path_residues(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs: object,
    ) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
        del y_train
        captured_kwargs.append(kwargs)
        alphas = np.array([1.0, 0.0], dtype=np.float64)
        active: list[int] = []
        coefs = np.zeros((X_train.shape[1], alphas.shape[0]), dtype=np.float64)
        residues = np.zeros((alphas.shape[0], y_test.shape[0]), dtype=np.float64)
        return alphas, active, coefs, residues

    def fake_fit(self: LarsCV, X_arg: np.ndarray, y_arg: np.ndarray, **kwargs: Any) -> None:
        del self, X_arg, y_arg, kwargs
        return None

    monkeypatch.setattr(least_angle, "_lars_path_residues", fake_lars_path_residues)
    monkeypatch.setattr(LarsCV, "_fit", fake_fit, raising=False)

    estimator.fit(X, y)

    expected = lars_cv_path_residues_callback_kwargs(
        precompute=precompute,
        method=method,
        verbose=verbose,
        fit_intercept=estimator.fit_intercept,
        max_iter=estimator.max_iter,
        eps=estimator.eps,
        positive=positive,
    )
    assert captured_kwargs
    assert all(payload == expected for payload in captured_kwargs)


def test_lars_cv_path_residues_callback_kwargs_preserve_precompute_identity() -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv_orchestration_shell import (
        lars_cv_path_residues_callback_kwargs,
    )

    precompute = object()

    payload = lars_cv_path_residues_callback_kwargs(
        precompute=precompute,
        method="lar",
        verbose=True,
        fit_intercept=False,
        max_iter=7,
        eps=1e-8,
    )

    assert payload["Gram"] is precompute
    assert payload["copy"] is False
    assert payload["verbose"] == 0
    assert payload["fit_intercept"] is False


def test_lars_cv_path_residues_callback_kwargs_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv_orchestration_shell import (
        lars_cv_path_residues_callback_kwargs,
    )

    with pytest.raises(ViolationError):
        lars_cv_path_residues_callback_kwargs(precompute=True, method="bad")

    with pytest.raises(ViolationError):
        lars_cv_path_residues_callback_kwargs(precompute=True, method="lar", verbose=-1)

    with pytest.raises(ViolationError):
        lars_cv_path_residues_callback_kwargs(precompute=True, method="lar", fit_intercept=1)

    with pytest.raises(ViolationError):
        lars_cv_path_residues_callback_kwargs(precompute=True, method="lar", max_iter=0)

    with pytest.raises(ViolationError):
        lars_cv_path_residues_callback_kwargs(precompute=True, method="lar", eps=np.nan)

    with pytest.raises(ViolationError):
        lars_cv_path_residues_callback_kwargs(precompute=True, method="lar", positive=1)
