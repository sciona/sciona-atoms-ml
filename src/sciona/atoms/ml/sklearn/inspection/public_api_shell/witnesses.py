"""Ghost witnesses for public sklearn inspection API-shell atoms."""

from __future__ import annotations


def witness_inspection_public_function_catalog(catalog_scope: str = "public_functions") -> tuple[str, ...]:
    """Describe public inspection functions covered by this shell."""
    del catalog_scope
    return ("partial_dependence", "permutation_importance")


def witness_inspection_public_function_boundary(function_name: str) -> str:
    """Describe the callback boundary behind a public inspection function."""
    if function_name == "partial_dependence":
        return "estimator_response_and_grid_callbacks"
    return "scorer_shuffle_and_joblib_callbacks"


def witness_partial_dependence_call_payload(
    estimator: object,
    X: object,
    features: object,
    *,
    sample_weight: object = None,
    categorical_features: object = None,
    feature_names: object = None,
    response_method: str = "auto",
    percentiles: tuple[float, float] = (0.05, 0.95),
    grid_resolution: int = 100,
    custom_values: object = None,
    method: str = "auto",
    kind: str = "average",
) -> dict[str, object]:
    """Describe a public partial_dependence call payload."""
    kwargs = {
        "sample_weight": sample_weight,
        "categorical_features": categorical_features,
        "feature_names": feature_names,
        "response_method": response_method,
        "percentiles": percentiles,
        "grid_resolution": grid_resolution,
        "custom_values": custom_values,
        "method": method,
        "kind": kind,
    }
    return {"function_name": "partial_dependence", "args": (estimator, X, features), "kwargs": kwargs}


def witness_permutation_importance_call_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    scoring: object = None,
    n_repeats: int = 5,
    n_jobs: int | None = None,
    random_state: int | None = None,
    sample_weight: object = None,
    max_samples: int | float = 1.0,
) -> dict[str, object]:
    """Describe a public permutation_importance call payload."""
    kwargs = {
        "scoring": scoring,
        "n_repeats": n_repeats,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "sample_weight": sample_weight,
        "max_samples": max_samples,
    }
    return {"function_name": "permutation_importance", "args": (estimator, X, y), "kwargs": kwargs}


def witness_partial_dependence_result_summary(inspection_result: object) -> dict[str, object]:
    """Describe public partial_dependence result metadata."""
    return {"inspection_result": inspection_result}


def witness_permutation_importance_result_summary(inspection_result: object) -> dict[str, object]:
    """Describe public permutation_importance result metadata."""
    return {"inspection_result": inspection_result}
