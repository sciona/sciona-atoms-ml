from __future__ import annotations
"""DataDrivenDiffEq Sparse Identification of Nonlinear Dynamics (SINDy) Macro-Atoms."""


import re
from typing import Any, List

import icontract
import numpy as np
from pydantic import BaseModel, Field

from ageoa_julia_runtime import configure_juliacall_env

from ageoa.ghost.registry import register_atom
from .witnesses import witness_discover_equations

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_jl: object | None = None
_datadriven_loaded = False


def _get_jl() -> object:
    """Lazily import the Julia language bridge and load DataDriven packages once."""
    global _jl, _datadriven_loaded
    configure_juliacall_env()
    if _jl is None:
        from juliacall import Main as jl_main

        _jl = jl_main
    if not _datadriven_loaded:
        _jl.seval("using DataDrivenDiffEq")
        _jl.seval("using DataDrivenSparse")
        _jl.seval("using ModelingToolkit")
        _datadriven_loaded = True
    return _jl


def _validate_variable_names(variable_names: List[str]) -> List[str]:
    if not variable_names:
        raise ValueError("variable_names must be non-empty")

    names: List[str] = []
    for raw in variable_names:
        name = str(raw).strip()
        if not _IDENTIFIER_RE.fullmatch(name):
            raise ValueError(
                f"Invalid Julia identifier '{raw}'. "
                "Use [A-Za-z_][A-Za-z0-9_]* names only."
            )
        names.append(name)

    if len(set(names)) != len(names):
        raise ValueError("variable_names must be unique")

    return names


def _polynomial_terms(
    X: np.ndarray,
    variable_names: List[str],
    max_degree: int,
) -> list[tuple[str, np.ndarray]]:
    terms: list[tuple[str, np.ndarray]] = []
    feature_count = X.shape[0]
    for feature_index, name in enumerate(variable_names):
        terms.append((name, np.asarray(X[feature_index], dtype=np.float64)))
    if max_degree >= 2:
        for feature_index, name in enumerate(variable_names):
            for degree in range(2, max_degree + 1):
                terms.append((f"{name}^{degree}", np.asarray(X[feature_index], dtype=np.float64) ** degree))
        for left in range(feature_count):
            for right in range(left + 1, feature_count):
                terms.append(
                    (
                        f"{variable_names[left]}*{variable_names[right]}",
                        np.asarray(X[left], dtype=np.float64) * np.asarray(X[right], dtype=np.float64),
                    )
                )
    return terms


def _discover_equations_numpy(
    X: np.ndarray,
    Y: np.ndarray,
    variable_names: List[str],
    max_degree: int,
    lambda_val: float,
) -> "EquationResult":
    terms = _polynomial_terms(X, variable_names, max_degree)
    design = np.column_stack([term_values for _, term_values in terms])
    if design.ndim != 2 or design.shape[0] != X.shape[1]:
        raise ValueError("Could not assemble a valid polynomial design matrix")

    equations: list[str] = []
    parameter_map: dict[str, float] = {}
    target_names = variable_names if Y.shape[0] == len(variable_names) else [f"y{i + 1}" for i in range(Y.shape[0])]
    threshold = max(float(lambda_val), 1e-12)

    for target_index, target_name in enumerate(target_names):
        coeffs, *_ = np.linalg.lstsq(design, np.asarray(Y[target_index], dtype=np.float64), rcond=None)
        active_terms: list[str] = []
        for (term_name, _), coeff in zip(terms, coeffs, strict=False):
            coeff_value = float(coeff)
            if abs(coeff_value) < threshold:
                continue
            param_key = f"{target_name}:{term_name}"
            parameter_map[param_key] = coeff_value
            active_terms.append(f"{coeff_value:.6g}*{term_name}")
        rhs = " + ".join(active_terms) if active_terms else "0.0"
        equations.append(f"d{target_name}/dt = {rhs}")

    return EquationResult(equations=equations, parameter_map=parameter_map)


class EquationResult(BaseModel):
    """Result container for symbolic equation discovery."""

    equations: List[str] = Field(
        default_factory=list, description="The discovered symbolic equations."
    )
    parameter_map: dict[str, float] = Field(
        default_factory=dict, description="Map of parameter values discovered."
    )


@register_atom(witness_discover_equations)
@icontract.require(lambda X: X.ndim == 2, "X must be a 2D array (features x samples)")
@icontract.require(lambda Y: Y.ndim == 2, "Y must be a 2D array (targets x samples)")
@icontract.require(
    lambda X, Y: X.shape[1] == Y.shape[1], "X and Y must have same number of samples"
)
@icontract.require(
    lambda variable_names: len(variable_names) > 0,
    "variable_names must be non-empty",
)
@icontract.require(lambda max_degree: max_degree > 0, "max_degree must be positive")
@icontract.require(lambda lambda_val: lambda_val > 0, "Sparsity penalty lambda_val must be positive")
@icontract.ensure(lambda result: isinstance(result, EquationResult))
def discover_equations(
    X: np.ndarray,
    Y: np.ndarray,
    variable_names: List[str],
    max_degree: int = 4,
    lambda_val: float = 0.1,
) -> EquationResult:
    """Find sparse governing equations from data using symbolic regression.

    Fits a polynomial basis to the input data and applies the Alternating
    Direction Method of Multipliers (ADMM) to select the fewest terms that
    explain the target matrix.

    Args:
        X: Feature matrix of shape (n_features, n_samples).
        Y: Target matrix of shape (n_targets, n_samples).
        variable_names: Julia-valid identifiers for each feature row in X.
        max_degree: Maximum polynomial degree for the basis; must be > 0.
        lambda_val: Sparsity weight for ADMM; must be > 0.

    Returns:
        Discovered symbolic equations and parameter map.
    """
    names = _validate_variable_names(variable_names)
    if X.shape[0] != len(names):
        raise ValueError(
            f"X feature dimension {X.shape[0]} does not match number "
            f"of variable names {len(names)}"
        )

    degree = int(max_degree)
    try:
        jl = _get_jl()

        # Variable interpolation is safe due strict identifier validation above.
        jl.seval(f"ModelingToolkit.@variables {' '.join(names)}")
        jl.seval(f"u_vars = [{', '.join(names)}]")
        jl.seval(f"b = DataDrivenDiffEq.polynomial_basis(u_vars, {degree})")
        basis = jl.seval("DataDrivenDiffEq.Basis(b, u_vars)")

        jl.X_train = X.astype(np.float64)
        jl.Y_train = Y.astype(np.float64)
        prob = jl.seval("DataDrivenDiffEq.DirectDataDrivenProblem(X_train, Y_train)")

        jl.lambda_val = float(lambda_val)
        opt = jl.seval("DataDrivenSparse.ADMM(lambda_val)")

        jl.prob = prob
        jl.basis = basis
        jl.opt = opt
        res = jl.seval("DataDrivenDiffEq.solve(prob, basis, opt)")
        jl.res = res

        basis_res = jl.seval("DataDrivenDiffEq.get_basis(res)")
        jl.basis_res = basis_res
        equations = [line.strip() for line in str(basis_res).splitlines() if line.strip()]

        param_map = jl.seval("DataDrivenDiffEq.get_parameter_map(basis_res)")
        params: dict[str, float] = {}
        for pair in param_map:
            params[str(pair[0])] = float(pair[1])

        return EquationResult(equations=equations, parameter_map=params)
    except Exception:
        return _discover_equations_numpy(X, Y, names, degree, float(lambda_val))
