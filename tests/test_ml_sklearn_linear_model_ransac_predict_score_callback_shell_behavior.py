from __future__ import annotations

import pytest
from icontract import ViolationError


def test_ransac_predict_score_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_predict_score_callback_shell import (
        ransac_public_nonrouting_params,
        ransac_public_predict_callback_payload,
        ransac_public_score_callback_payload,
        ransac_public_validation_kwargs,
    )

    assert callable(ransac_public_validation_kwargs)
    assert callable(ransac_public_nonrouting_params)
    assert callable(ransac_public_predict_callback_payload)
    assert callable(ransac_public_score_callback_payload)


def test_ransac_public_validation_kwargs_match_predict_and_score_source() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_predict_score_callback_shell import ransac_public_validation_kwargs

    assert ransac_public_validation_kwargs("predict") == {
        "ensure_all_finite": False,
        "accept_sparse": True,
        "reset": False,
    }
    assert ransac_public_validation_kwargs("score") == {
        "ensure_all_finite": False,
        "accept_sparse": True,
        "reset": False,
    }


def test_ransac_public_nonrouting_params_match_empty_fallback() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_predict_score_callback_shell import ransac_public_nonrouting_params

    assert ransac_public_nonrouting_params("predict") == {}
    assert ransac_public_nonrouting_params("score") == {}


def test_ransac_public_predict_callback_payload_preserves_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_predict_score_callback_shell import ransac_public_predict_callback_payload

    estimator = object()
    X = object()
    predict_params = {"return_std": True}

    payload = ransac_public_predict_callback_payload(estimator, X, predict_params)

    assert payload == {"estimator": estimator, "X": X, "predict_params": predict_params}
    assert payload["estimator"] is estimator
    assert payload["X"] is X
    assert payload["predict_params"] is predict_params


def test_ransac_public_score_callback_payload_preserves_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_predict_score_callback_shell import ransac_public_score_callback_payload

    estimator = object()
    X = object()
    y = object()
    score_params = {"sample_weight": object()}

    payload = ransac_public_score_callback_payload(estimator, X, y, score_params)

    assert payload == {"estimator": estimator, "X": X, "y": y, "score_params": score_params}
    assert payload["estimator"] is estimator
    assert payload["X"] is X
    assert payload["y"] is y
    assert payload["score_params"] is score_params


def test_ransac_predict_score_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_predict_score_callback_shell import (
        ransac_public_nonrouting_params,
        ransac_public_predict_callback_payload,
        ransac_public_score_callback_payload,
        ransac_public_validation_kwargs,
    )

    with pytest.raises(ViolationError):
        ransac_public_validation_kwargs("fit")

    with pytest.raises(ViolationError):
        ransac_public_nonrouting_params("transform")

    with pytest.raises(ViolationError):
        ransac_public_predict_callback_payload(object(), object(), None)

    with pytest.raises(ViolationError):
        ransac_public_score_callback_payload(object(), object(), object(), None)
