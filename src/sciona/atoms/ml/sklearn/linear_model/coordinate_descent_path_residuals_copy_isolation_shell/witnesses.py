"""Ghost witnesses for coordinate-descent path-residual copy-isolation atoms."""

from __future__ import annotations

from collections.abc import Mapping


def witness_cd_path_residuals_path_params_copy(path_params: Mapping[object, object]) -> dict[object, object]:
    """Describe the shallow path_params.copy() isolation before local mutation."""
    return dict(path_params)
