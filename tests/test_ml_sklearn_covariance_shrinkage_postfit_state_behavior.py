from __future__ import annotations

import numpy as np
import pytest
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance


def test_covariance_shrinkage_postfit_state_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.covariance_shrinkage_postfit_state import (
        covariance_fit_return_self,
        covariance_ledoit_wolf_fit_shrinkage,
        covariance_oas_fit_shrinkage,
    )

    assert callable(covariance_ledoit_wolf_fit_shrinkage)
    assert callable(covariance_oas_fit_shrinkage)
    assert callable(covariance_fit_return_self)


def test_covariance_shrinkage_postfit_state_matches_sklearn_estimators() -> None:
    from sciona.atoms.ml.sklearn.covariance.covariance_shrinkage_postfit_state import (
        covariance_fit_return_self,
        covariance_ledoit_wolf_fit_shrinkage,
        covariance_oas_fit_shrinkage,
    )

    X = np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 2.0, 4.0],
            [2.0, 4.0, 7.0],
            [4.0, 8.0, 9.0],
            [5.0, 9.0, 12.0],
        ],
        dtype=np.float64,
    )

    ledoit = LedoitWolf(block_size=2).fit(X)
    oas = OAS().fit(X)

    assert covariance_ledoit_wolf_fit_shrinkage(ledoit.shrinkage_) == pytest.approx(ledoit.shrinkage_)
    assert covariance_oas_fit_shrinkage(oas.shrinkage_) == pytest.approx(oas.shrinkage_)

    for token in ["EmpiricalCovariance", "ShrunkCovariance", "LedoitWolf", "OAS"]:
        assert covariance_fit_return_self(token) == token

    assert isinstance(EmpiricalCovariance().fit(X), EmpiricalCovariance)
    assert isinstance(ShrunkCovariance().fit(X), ShrunkCovariance)
    assert isinstance(ledoit, LedoitWolf)
    assert isinstance(oas, OAS)


def test_covariance_shrinkage_postfit_state_contracts() -> None:
    from sciona.atoms.ml.sklearn.covariance.covariance_shrinkage_postfit_state import (
        covariance_fit_return_self,
        covariance_ledoit_wolf_fit_shrinkage,
        covariance_oas_fit_shrinkage,
    )

    with pytest.raises(Exception):
        covariance_ledoit_wolf_fit_shrinkage(-0.1)

    with pytest.raises(Exception):
        covariance_oas_fit_shrinkage(1.1)

    with pytest.raises(Exception):
        covariance_fit_return_self("")
