from __future__ import annotations

import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.feature_selection.select_from_model_postfit_attributes import (
    select_from_model_partial_fit_first_call,
    select_from_model_postfit_feature_names_in,
    select_from_model_postfit_n_features_in,
)


def test_select_from_model_partial_fit_first_call_matches_hasattr_logic() -> None:
    assert select_from_model_partial_fit_first_call(estimator_is_initialized=False) is True
    assert select_from_model_partial_fit_first_call(estimator_is_initialized=True) is False


def test_select_from_model_postfit_attributes_pass_through() -> None:
    assert select_from_model_postfit_n_features_in(4) == 4
    assert select_from_model_postfit_feature_names_in(("x0", "x1")) == ("x0", "x1")


def test_select_from_model_postfit_attributes_reject_invalid_inputs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        select_from_model_partial_fit_first_call(estimator_is_initialized="nope")  # type: ignore[arg-type]

    with pytest.raises((ViolationError, ValueError)):
        select_from_model_postfit_n_features_in(0)

    with pytest.raises((ViolationError, ValueError)):
        select_from_model_postfit_feature_names_in(("x0", ""))
