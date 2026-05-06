from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.utils import Bunch


def test_coordinate_descent_cv_nonrouting_fallback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_nonrouting_fallback_shell import (
        cd_cv_nonrouting_empty_split_params,
        cd_cv_nonrouting_routed_params,
        cd_cv_nonrouting_split_kwargs,
        cd_cv_nonrouting_splitter_payload,
    )

    assert callable(cd_cv_nonrouting_empty_split_params)
    assert callable(cd_cv_nonrouting_splitter_payload)
    assert callable(cd_cv_nonrouting_routed_params)
    assert callable(cd_cv_nonrouting_split_kwargs)


def test_coordinate_descent_cv_nonrouting_fallback_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_nonrouting_fallback_shell import (
        cd_cv_nonrouting_empty_split_params,
        cd_cv_nonrouting_routed_params,
        cd_cv_nonrouting_split_kwargs,
        cd_cv_nonrouting_splitter_payload,
    )

    split_params = cd_cv_nonrouting_empty_split_params(True)
    assert isinstance(split_params, Bunch)
    assert dict(split_params) == {}

    splitter_payload = cd_cv_nonrouting_splitter_payload(split_params)
    assert isinstance(splitter_payload, Bunch)
    assert splitter_payload.split is split_params

    routed_params = cd_cv_nonrouting_routed_params(True, splitter_payload)
    assert isinstance(routed_params, Bunch)
    assert routed_params.splitter is splitter_payload
    assert cd_cv_nonrouting_split_kwargs(routed_params) is split_params


def test_coordinate_descent_cv_nonrouting_fallback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_nonrouting_fallback_shell import (
        cd_cv_nonrouting_empty_split_params,
        cd_cv_nonrouting_routed_params,
        cd_cv_nonrouting_split_kwargs,
        cd_cv_nonrouting_splitter_payload,
    )

    with pytest.raises(ViolationError):
        cd_cv_nonrouting_empty_split_params(False)

    with pytest.raises(ViolationError):
        cd_cv_nonrouting_splitter_payload({})

    with pytest.raises(ViolationError):
        cd_cv_nonrouting_routed_params(True, {})

    with pytest.raises(ViolationError):
        cd_cv_nonrouting_split_kwargs(Bunch())
