"""Witnesses for sklearn multioutput response-method bookkeeping helpers."""

from __future__ import annotations


def witness_response_method_candidates(response_method: str | tuple[str, ...]) -> tuple[str, ...]:
    """Describe response-method candidate normalization."""
    if isinstance(response_method, str):
        return (response_method,)
    return tuple(response_method)


def witness_response_method_name(
    method_candidates: tuple[str, ...],
    implemented_methods: tuple[str, ...],
    *,
    estimator_name: str,
) -> str:
    """Describe first-available response-method selection."""
    for method_name in method_candidates:
        if method_name in implemented_methods:
            return method_name
    raise AttributeError(
        f"{estimator_name} has none of the following attributes: {', '.join(method_candidates)}."
    )


def witness_chain_fit_chain_method_name(
    chain_method: str | tuple[str, ...],
    implemented_methods: tuple[str, ...],
    *,
    estimator_name: str,
) -> str:
    """Describe chain-method name resolution during chain fit."""
    return witness_response_method_name(
        witness_response_method_candidates(chain_method),
        implemented_methods,
        estimator_name=estimator_name,
    )


def witness_multioutput_predict_proba_available_from_estimators(
    fitted_predict_proba_flags: tuple[bool, ...],
) -> bool:
    """Describe fitted-estimator predict_proba availability checking."""
    if not all(fitted_predict_proba_flags):
        raise AttributeError("One or more fitted estimators do not implement predict_proba.")
    return True


def witness_multioutput_predict_proba_available_from_base_estimator(
    base_estimator_has_predict_proba: bool,
) -> bool:
    """Describe base-estimator predict_proba availability checking."""
    if not base_estimator_has_predict_proba:
        raise AttributeError("Base estimator does not implement predict_proba.")
    return True
