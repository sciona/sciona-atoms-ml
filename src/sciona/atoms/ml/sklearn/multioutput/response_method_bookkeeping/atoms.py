"""Multioutput response-method bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_chain_fit_chain_method_name,
    witness_multioutput_predict_proba_available_from_base_estimator,
    witness_multioutput_predict_proba_available_from_estimators,
    witness_response_method_candidates,
    witness_response_method_name,
)

VALID_RESPONSE_METHODS = {
    "predict",
    "predict_proba",
    "predict_log_proba",
    "decision_function",
}

MethodSpec = str | tuple[str, ...]


def _flag_valid(value: object) -> bool:
    return isinstance(value, bool)


def _method_name_valid(value: object) -> bool:
    return isinstance(value, str) and value in VALID_RESPONSE_METHODS


def _method_spec_valid(value: object) -> bool:
    return bool(
        _method_name_valid(value)
        or (
            isinstance(value, tuple)
            and len(value) >= 1
            and all(_method_name_valid(item) for item in value)
        )
    )


def _method_candidates_valid(value: object) -> bool:
    return bool(
        isinstance(value, tuple)
        and len(value) >= 1
        and all(_method_name_valid(item) for item in value)
    )


def _implemented_methods_valid(value: object) -> bool:
    return bool(
        isinstance(value, tuple)
        and all(_method_name_valid(item) for item in value)
    )


def _estimator_name_valid(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


@register_atom(witness_response_method_candidates)
@icontract.require(lambda response_method: _method_spec_valid(response_method), "response_method must be a supported method name or nonempty tuple of supported method names")
@icontract.ensure(lambda result: _method_candidates_valid(result), "result must be a nonempty tuple of supported method names")
def response_method_candidates(response_method: MethodSpec) -> tuple[str, ...]:
    """Normalize sklearn response_method input into an ordered candidate tuple."""
    if isinstance(response_method, str):
        return (response_method,)
    return tuple(response_method)


@register_atom(witness_response_method_name)
@icontract.require(lambda method_candidates: _method_candidates_valid(method_candidates), "method_candidates must be a nonempty tuple of supported method names")
@icontract.require(lambda implemented_methods: _implemented_methods_valid(implemented_methods), "implemented_methods must be a tuple of supported method names")
@icontract.require(lambda estimator_name: _estimator_name_valid(estimator_name), "estimator_name must be a nonempty string")
@icontract.ensure(lambda result: _method_name_valid(result), "result must be a supported method name")
def response_method_name(
    method_candidates: tuple[str, ...],
    implemented_methods: tuple[str, ...],
    *,
    estimator_name: str,
) -> str:
    """Select sklearn's first implemented response method from an ordered candidate tuple."""
    for method_name in method_candidates:
        if method_name in implemented_methods:
            return method_name
    raise AttributeError(
        f"{estimator_name} has none of the following attributes: {', '.join(method_candidates)}."
    )


@register_atom(witness_chain_fit_chain_method_name)
@icontract.require(lambda chain_method: _method_spec_valid(chain_method), "chain_method must be a supported method name or nonempty tuple of supported method names")
@icontract.require(lambda implemented_methods: _implemented_methods_valid(implemented_methods), "implemented_methods must be a tuple of supported method names")
@icontract.require(lambda estimator_name: _estimator_name_valid(estimator_name), "estimator_name must be a nonempty string")
@icontract.ensure(lambda result: _method_name_valid(result), "result must be a supported chain method name")
def chain_fit_chain_method_name(
    chain_method: MethodSpec,
    implemented_methods: tuple[str, ...],
    *,
    estimator_name: str,
) -> str:
    """Resolve the chain_method name sklearn stores during chain fit."""
    candidates = response_method_candidates(chain_method)
    return response_method_name(
        candidates,
        implemented_methods,
        estimator_name=estimator_name,
    )


@register_atom(witness_multioutput_predict_proba_available_from_estimators)
@icontract.require(
    lambda fitted_predict_proba_flags: isinstance(fitted_predict_proba_flags, tuple)
    and len(fitted_predict_proba_flags) >= 1
    and all(_flag_valid(flag) for flag in fitted_predict_proba_flags),
    "fitted_predict_proba_flags must be a nonempty tuple of booleans",
)
@icontract.ensure(lambda result: _flag_valid(result), "result must be boolean")
def multioutput_predict_proba_available_from_estimators(
    fitted_predict_proba_flags: tuple[bool, ...],
) -> bool:
    """Require sklearn's fitted-estimator predict_proba availability before multioutput predict_proba proceeds."""
    if not all(fitted_predict_proba_flags):
        raise AttributeError("One or more fitted estimators do not implement predict_proba.")
    return True


@register_atom(witness_multioutput_predict_proba_available_from_base_estimator)
@icontract.require(lambda base_estimator_has_predict_proba: _flag_valid(base_estimator_has_predict_proba), "base_estimator_has_predict_proba must be boolean")
@icontract.ensure(lambda result: _flag_valid(result), "result must be boolean")
def multioutput_predict_proba_available_from_base_estimator(
    base_estimator_has_predict_proba: bool,
) -> bool:
    """Require sklearn's unfitted base-estimator predict_proba availability for MultiOutputClassifier."""
    if not base_estimator_has_predict_proba:
        raise AttributeError("Base estimator does not implement predict_proba.")
    return True
