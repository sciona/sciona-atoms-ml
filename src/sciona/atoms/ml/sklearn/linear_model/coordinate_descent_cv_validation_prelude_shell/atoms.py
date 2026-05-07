"""Sklearn coordinate-descent CV validation-prelude shell atoms."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_check_y_params,
    witness_cd_cv_fit_params_guard_args,
    witness_cd_cv_fortran_check_x_params,
    witness_cd_cv_initial_copy_x,
    witness_cd_cv_non_reference_copy_x,
    witness_cd_cv_reference_check_x_params,
    witness_cd_cv_reference_validation_copy_x,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_cd_cv_fit_params_guard_args)
@icontract.require(lambda params: isinstance(params, dict), "params must be a dict")
@icontract.ensure(
    lambda result, params, estimator: isinstance(result, tuple)
    and len(result) == 3
    and result[0] == params
    and result[1] is estimator
    and result[2] == "fit",
    "fit params guard args must match _raise_for_params(params, self, 'fit')",
)
def cd_cv_fit_params_guard_args(
    params: dict[object, object], estimator: object
) -> tuple[dict[object, object], object, str]:
    """Return positional args for _raise_for_params(params, self, 'fit')."""
    return (dict(params), estimator, "fit")


@register_atom(witness_cd_cv_initial_copy_x)
@icontract.require(lambda copy_x: _bool(copy_x), "copy_x must be boolean")
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.ensure(
    lambda result, copy_x, fit_intercept: _bool(result)
    and result == (copy_x and fit_intercept),
    "initial copy_X must match self.copy_X and self.fit_intercept",
)
def cd_cv_initial_copy_x(copy_x: bool, fit_intercept: bool) -> bool:
    """Return the initial copy_X value used before validation."""
    return copy_x and fit_intercept


@register_atom(witness_cd_cv_check_y_params)
@icontract.require(lambda validation_required: validation_required is True, "validation must be required")
@icontract.ensure(
    lambda result: isinstance(result, dict)
    and result == {
        "copy": False,
        "dtype": [np.float64, np.float32],
        "ensure_2d": False,
    },
    "check_y_params must match LinearModelCV.fit",
)
def cd_cv_check_y_params(validation_required: bool) -> dict[str, object]:
    """Return the y-validation kwargs used by LinearModelCV.fit."""
    del validation_required
    return {"copy": False, "dtype": [np.float64, np.float32], "ensure_2d": False}


@register_atom(witness_cd_cv_reference_check_x_params)
@icontract.require(
    lambda reference_branch_required: reference_branch_required is True,
    "reference-preserving validation branch must be required",
)
@icontract.ensure(
    lambda result: isinstance(result, dict)
    and result == {
        "accept_sparse": "csc",
        "dtype": [np.float64, np.float32],
        "force_writeable": True,
        "copy": False,
        "accept_large_sparse": False,
    },
    "reference-preserving check_X_params must match LinearModelCV.fit",
)
def cd_cv_reference_check_x_params(reference_branch_required: bool) -> dict[str, object]:
    """Return X-validation kwargs for ndarray or sparse inputs."""
    del reference_branch_required
    return {
        "accept_sparse": "csc",
        "dtype": [np.float64, np.float32],
        "force_writeable": True,
        "copy": False,
        "accept_large_sparse": False,
    }


@register_atom(witness_cd_cv_fortran_check_x_params)
@icontract.require(lambda copy_x: _bool(copy_x), "copy_x must be boolean")
@icontract.require(
    lambda reference_branch_required: reference_branch_required is False,
    "reference-preserving validation branch must be skipped",
)
@icontract.ensure(
    lambda result, copy_x: isinstance(result, dict)
    and result == {
        "accept_sparse": "csc",
        "dtype": [np.float64, np.float32],
        "order": "F",
        "force_writeable": True,
        "copy": copy_x,
    },
    "Fortran-order check_X_params must match LinearModelCV.fit",
)
def cd_cv_fortran_check_x_params(
    copy_x: bool, reference_branch_required: bool
) -> dict[str, object]:
    """Return X-validation kwargs for non-ndarray, non-sparse inputs."""
    del reference_branch_required
    return {
        "accept_sparse": "csc",
        "dtype": [np.float64, np.float32],
        "order": "F",
        "force_writeable": True,
        "copy": copy_x,
    }


@register_atom(witness_cd_cv_reference_validation_copy_x)
@icontract.require(lambda copy_x: _bool(copy_x), "copy_x must be boolean")
@icontract.require(lambda x_is_sparse: _bool(x_is_sparse), "x_is_sparse must be boolean")
@icontract.require(lambda sparse_data_copied: _bool(sparse_data_copied), "sparse_data_copied must be boolean")
@icontract.require(lambda dense_array_copied: _bool(dense_array_copied), "dense_array_copied must be boolean")
@icontract.ensure(
    lambda result, copy_x, x_is_sparse, sparse_data_copied, dense_array_copied: _bool(result)
    and result
    == (
        False
        if ((x_is_sparse and sparse_data_copied) or ((not x_is_sparse) and dense_array_copied))
        else copy_x
    ),
    "reference-branch copy_X must reset only when validation copied X",
)
def cd_cv_reference_validation_copy_x(
    copy_x: bool,
    x_is_sparse: bool,
    sparse_data_copied: bool,
    dense_array_copied: bool,
) -> bool:
    """Return copy_X after reference-preserving validation copy checks."""
    if (x_is_sparse and sparse_data_copied) or ((not x_is_sparse) and dense_array_copied):
        return False
    return copy_x


@register_atom(witness_cd_cv_non_reference_copy_x)
@icontract.require(
    lambda reference_branch_required: reference_branch_required is False,
    "reference-preserving validation branch must be skipped",
)
@icontract.ensure(lambda result: result is False, "non-reference validation branch must reset copy_X to False")
def cd_cv_non_reference_copy_x(reference_branch_required: bool) -> bool:
    """Return copy_X after non-reference Fortran-order validation."""
    del reference_branch_required
    return False
