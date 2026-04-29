from __future__ import annotations

import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.multioutput.response_method_bookkeeping import (
    chain_fit_chain_method_name,
    multioutput_predict_proba_available_from_base_estimator,
    multioutput_predict_proba_available_from_estimators,
    response_method_candidates,
    response_method_name,
)


def test_response_method_candidates_normalizes_string_and_tuple() -> None:
    assert response_method_candidates("predict") == ("predict",)
    assert response_method_candidates(("decision_function", "predict_proba")) == (
        "decision_function",
        "predict_proba",
    )


def test_response_method_name_selects_first_available_method() -> None:
    assert response_method_name(
        ("decision_function", "predict_proba"),
        ("predict", "predict_proba"),
        estimator_name="DummyEstimator",
    ) == "predict_proba"


def test_chain_fit_chain_method_name_matches_classifier_chain_resolution() -> None:
    assert chain_fit_chain_method_name(
        ("predict_proba", "decision_function", "predict"),
        ("predict", "decision_function"),
        estimator_name="DummyClassifier",
    ) == "decision_function"


def test_response_method_resolution_raises_when_no_candidate_is_implemented() -> None:
    with pytest.raises(AttributeError, match="DummyEstimator has none of the following attributes"):
        response_method_name(
            ("predict_log_proba", "predict_proba"),
            ("predict",),
            estimator_name="DummyEstimator",
        )


def test_multioutput_predict_proba_availability_guards_match_shell_checks() -> None:
    assert multioutput_predict_proba_available_from_estimators((True, True)) is True
    assert multioutput_predict_proba_available_from_base_estimator(True) is True

    with pytest.raises(AttributeError, match="fitted estimators do not implement predict_proba"):
        multioutput_predict_proba_available_from_estimators((True, False))

    with pytest.raises(AttributeError, match="Base estimator does not implement predict_proba"):
        multioutput_predict_proba_available_from_base_estimator(False)


def test_response_method_bookkeeping_rejects_invalid_specs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        response_method_candidates(("predict", "nope"))
