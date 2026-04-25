from __future__ import annotations

import numpy as np
import pytest
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def test_partial_dependence_preflight_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_preflight import (
        partial_dependence_require_no_sample_weight_for_recursion,
        partial_dependence_require_recursion_support,
        partial_dependence_require_response_method_auto_for_regressor,
        partial_dependence_resolve_auto_method,
        partial_dependence_resolve_kind_method,
    )

    assert callable(partial_dependence_require_response_method_auto_for_regressor)
    assert callable(partial_dependence_resolve_kind_method)
    assert callable(partial_dependence_require_no_sample_weight_for_recursion)
    assert callable(partial_dependence_resolve_auto_method)
    assert callable(partial_dependence_require_recursion_support)


def test_partial_dependence_response_method_guard_matches_sklearn_regressor_contract() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_preflight import (
        partial_dependence_require_response_method_auto_for_regressor,
    )

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    est = LinearRegression().fit(X, y)

    with pytest.raises(ValueError, match="ignored for regressors and must be 'auto'"):
        partial_dependence_require_response_method_auto_for_regressor("regressor", "predict_proba")
    with pytest.raises(ValueError, match="ignored for regressors and must be 'auto'"):
        partial_dependence(est, X=X, features=[0], response_method="predict_proba")

    assert partial_dependence_require_response_method_auto_for_regressor("regressor", "auto") == "auto"
    assert partial_dependence_require_response_method_auto_for_regressor("classifier", "decision_function") == "decision_function"


def test_partial_dependence_kind_method_resolution_matches_sklearn_contract() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_preflight import partial_dependence_resolve_kind_method

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    est = DecisionTreeRegressor(random_state=0).fit(X, y)

    assert partial_dependence_resolve_kind_method("individual", "auto") == "brute"
    assert partial_dependence_resolve_kind_method("both", "brute") == "brute"
    assert partial_dependence_resolve_kind_method("average", "auto") == "auto"

    with pytest.raises(ValueError, match="only applies when 'kind' is set to 'average'"):
        partial_dependence_resolve_kind_method("individual", "recursion")
    with pytest.raises(ValueError, match="only applies when 'kind' is set to 'average'"):
        partial_dependence(est, X=X, features=[0], kind="individual", method="recursion")


def test_partial_dependence_sample_weight_guard_matches_sklearn_contract() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_preflight import (
        partial_dependence_require_no_sample_weight_for_recursion,
    )

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    sample_weight = np.ones(X.shape[0], dtype=np.float64)
    est = DecisionTreeRegressor(random_state=0).fit(X, y)

    assert partial_dependence_require_no_sample_weight_for_recursion("brute", sample_weight_provided=True) == "brute"

    with pytest.raises(ValueError, match="sample_weight is None"):
        partial_dependence_require_no_sample_weight_for_recursion("recursion", sample_weight_provided=True)
    with pytest.raises(ValueError, match="sample_weight is None"):
        partial_dependence(
            est,
            X=X,
            features=[0],
            method="recursion",
            sample_weight=sample_weight,
        )


def test_partial_dependence_auto_method_resolution_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_preflight import partial_dependence_resolve_auto_method

    assert partial_dependence_resolve_auto_method(
        "auto",
        sample_weight_provided=True,
        supports_recursion=True,
    ) == "brute"
    assert partial_dependence_resolve_auto_method(
        "auto",
        sample_weight_provided=False,
        supports_recursion=True,
    ) == "recursion"
    assert partial_dependence_resolve_auto_method(
        "auto",
        sample_weight_provided=False,
        supports_recursion=False,
    ) == "brute"
    assert partial_dependence_resolve_auto_method(
        "recursion",
        sample_weight_provided=False,
        supports_recursion=False,
    ) == "recursion"


def test_partial_dependence_recursion_support_guard_rejects_unsupported_mode() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_preflight import (
        partial_dependence_require_recursion_support,
    )

    with pytest.raises(ValueError, match="recursion-supported estimators"):
        partial_dependence_require_recursion_support("recursion", supports_recursion=False)

    assert partial_dependence_require_recursion_support("brute", supports_recursion=False) == "brute"
    assert partial_dependence_require_recursion_support("recursion", supports_recursion=True) == "recursion"


def test_partial_dependence_preflight_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_preflight import (
        partial_dependence_require_no_sample_weight_for_recursion,
        partial_dependence_require_recursion_support,
        partial_dependence_require_response_method_auto_for_regressor,
        partial_dependence_resolve_auto_method,
        partial_dependence_resolve_kind_method,
    )

    with pytest.raises(Exception):
        partial_dependence_require_response_method_auto_for_regressor("clusterer", "auto")
    with pytest.raises(Exception):
        partial_dependence_resolve_kind_method("summary", "auto")
    with pytest.raises(Exception):
        partial_dependence_require_no_sample_weight_for_recursion("auto", sample_weight_provided=np.bool_(True))
    with pytest.raises(Exception):
        partial_dependence_resolve_auto_method("magic", sample_weight_provided=False, supports_recursion=False)
    with pytest.raises(Exception):
        partial_dependence_require_recursion_support("auto", supports_recursion=np.bool_(False))
