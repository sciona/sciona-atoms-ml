from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_estimator_prefit_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_prefit_callback_shell import (
        cd_estimator_prefit_result_unpack,
        cd_estimator_prefit_xy_payload,
        cd_estimator_set_order_result_unpack,
    )

    assert callable(cd_estimator_prefit_result_unpack)
    assert callable(cd_estimator_set_order_result_unpack)
    assert callable(cd_estimator_prefit_xy_payload)


def test_coordinate_descent_estimator_prefit_callback_shell_matches_callback_unpacking() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_prefit_callback_shell import (
        cd_estimator_prefit_result_unpack,
        cd_estimator_prefit_xy_payload,
        cd_estimator_set_order_result_unpack,
    )

    X = object()
    y = object()
    X_offset = object()
    y_offset = object()
    X_scale = object()
    precompute = object()
    Xy = object()

    prefit = cd_estimator_prefit_result_unpack(
        (X, y, X_offset, y_offset, X_scale, precompute, Xy)
    )
    assert prefit == {
        "X": X,
        "y": y,
        "X_offset": X_offset,
        "y_offset": y_offset,
        "X_scale": X_scale,
        "precompute": precompute,
        "Xy": Xy,
    }

    ordered_X = object()
    ordered_y = object()
    assert cd_estimator_set_order_result_unpack((ordered_X, ordered_y)) == {
        "X": ordered_X,
        "y": ordered_y,
    }

    payload = cd_estimator_prefit_xy_payload(
        ordered_X, ordered_y, X_offset, y_offset, X_scale, precompute, Xy
    )
    assert list(payload) == ["X", "y", "X_offset", "y_offset", "X_scale", "precompute", "Xy"]
    assert tuple(payload.values()) == (
        ordered_X,
        ordered_y,
        X_offset,
        y_offset,
        X_scale,
        precompute,
        Xy,
    )


def test_coordinate_descent_estimator_prefit_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_prefit_callback_shell import (
        cd_estimator_prefit_result_unpack,
        cd_estimator_set_order_result_unpack,
    )

    with pytest.raises(ViolationError):
        cd_estimator_prefit_result_unpack((object(), object()))

    with pytest.raises(ViolationError):
        cd_estimator_prefit_result_unpack("not-a-prefit-result")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_estimator_set_order_result_unpack((object(), object(), object()))
