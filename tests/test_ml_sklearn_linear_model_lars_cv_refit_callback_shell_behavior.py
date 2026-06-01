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


def test_lars_cv_refit_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv_refit_callback_shell import (
        lars_cv_fit_return_self,
        lars_cv_refit_fit_call,
        lars_cv_refit_fit_kwargs,
        lars_cv_refit_state_payload,
    )

    assert callable(lars_cv_refit_state_payload)
    assert callable(lars_cv_refit_fit_kwargs)
    assert callable(lars_cv_refit_fit_call)
    assert callable(lars_cv_fit_return_self)


@pytest.mark.parametrize(
    "estimator",
    [
        LarsCV(cv=KFold(n_splits=3), max_iter=5, verbose=0, precompute=True),
        LassoLarsCV(cv=KFold(n_splits=3), max_iter=5, verbose=0, precompute=True),
    ],
)
def test_lars_cv_refit_callback_payload_matches_final_fit_call(
    monkeypatch: pytest.MonkeyPatch,
    estimator: LarsCV,
) -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv_refit_callback_shell import (
        lars_cv_fit_return_self,
        lars_cv_refit_fit_call,
        lars_cv_refit_fit_kwargs,
        lars_cv_refit_state_payload,
    )

    X, y = _make_regression_fixture()
    captured: dict[str, object] = {}

    def fake_lars_path_residues(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs: object,
    ) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
        del y_train, X_test, kwargs
        alphas = np.array([1.0, 0.5, 0.0], dtype=np.float64)
        active: list[int] = []
        coefs = np.zeros((X_train.shape[1], alphas.shape[0]), dtype=np.float64)
        residues = np.vstack(
            [
                np.full(y_test.shape[0], 1.5, dtype=np.float64),
                np.full(y_test.shape[0], 0.25, dtype=np.float64),
                np.full(y_test.shape[0], 1.0, dtype=np.float64),
            ]
        )
        return alphas, active, coefs, residues

    def fake_fit(self: LarsCV, X_arg: np.ndarray, y_arg: np.ndarray, **kwargs: Any) -> None:
        captured["self"] = self
        captured["X"] = X_arg
        captured["y"] = y_arg
        captured["kwargs"] = kwargs
        return None

    monkeypatch.setattr(least_angle, "_lars_path_residues", fake_lars_path_residues)
    monkeypatch.setattr(LarsCV, "_fit", fake_fit, raising=False)

    returned = estimator.fit(X, y)

    expected_kwargs = lars_cv_refit_fit_kwargs(estimator.max_iter, estimator.alpha_)
    assert captured["kwargs"] == expected_kwargs
    assert lars_cv_refit_fit_call(captured["X"], captured["y"], captured["kwargs"]) == {
        "args": (captured["X"], captured["y"]),
        "kwargs": captured["kwargs"],
    }
    state_payload = lars_cv_refit_state_payload(estimator.alpha_, estimator.cv_alphas_, estimator.mse_path_)
    assert state_payload["alpha_"] == estimator.alpha_
    assert state_payload["cv_alphas_"] is estimator.cv_alphas_
    assert state_payload["mse_path_"] is estimator.mse_path_
    assert lars_cv_fit_return_self(estimator) is estimator
    assert returned is estimator


def test_lars_cv_refit_state_payload_preserves_array_identities() -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv_refit_callback_shell import lars_cv_refit_state_payload

    cv_alphas = np.array([1.0, 0.5, 0.0], dtype=np.float64)
    mse_path = np.ones((3, 2), dtype=np.float64)

    payload = lars_cv_refit_state_payload(0.5, cv_alphas, mse_path)

    assert payload == {"alpha_": 0.5, "cv_alphas_": cv_alphas, "mse_path_": mse_path}
    assert payload["cv_alphas_"] is cv_alphas
    assert payload["mse_path_"] is mse_path


def test_lars_cv_refit_fit_kwargs_match_source_payload() -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv_refit_callback_shell import lars_cv_refit_fit_kwargs

    assert lars_cv_refit_fit_kwargs(17, np.float64(0.25)) == {
        "max_iter": 17,
        "alpha": 0.25,
        "Xy": None,
        "fit_path": True,
    }


def test_lars_cv_refit_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.lars_cv_refit_callback_shell import (
        lars_cv_fit_return_self,
        lars_cv_refit_fit_call,
        lars_cv_refit_fit_kwargs,
        lars_cv_refit_state_payload,
    )

    with pytest.raises(ViolationError):
        lars_cv_refit_state_payload(np.nan, object(), object())

    with pytest.raises(ViolationError):
        lars_cv_refit_fit_kwargs(0, 0.1)

    with pytest.raises(ViolationError):
        lars_cv_refit_fit_kwargs(10, -0.1)

    with pytest.raises(ViolationError):
        lars_cv_refit_fit_call(object(), object(), {"max_iter": 10, "alpha": 0.1})

    with pytest.raises(ViolationError):
        lars_cv_fit_return_self(None)
