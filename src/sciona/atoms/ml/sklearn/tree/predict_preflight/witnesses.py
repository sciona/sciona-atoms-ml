"""Ghost witnesses for sklearn tree prediction-preflight atoms."""

from __future__ import annotations


def witness_tree_predict_use_check_input_branch(check_input: bool) -> bool:
    """Describe the check_input branch predicate in _validate_X_predict."""
    return check_input


def witness_tree_predict_ensure_all_finite_mode(supports_missing_values: bool) -> str | bool:
    """Describe ensure_all_finite mode selection in _validate_X_predict."""
    return "allow-nan" if supports_missing_values else True


def witness_tree_predict_require_sparse_int32_indices(indices: object, indptr: object) -> bool:
    """Describe the CSR sparse-index dtype guard in _validate_X_predict."""
    del indices
    del indptr
    return True

