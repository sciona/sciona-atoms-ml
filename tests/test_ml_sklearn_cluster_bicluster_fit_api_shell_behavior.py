from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering


def test_bicluster_fit_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_fit_api_shell import (
        bicluster_fit_accept_sparse_format,
        bicluster_fit_dtype_name,
        bicluster_sparse_input_tag,
    )

    assert callable(bicluster_fit_accept_sparse_format)
    assert callable(bicluster_fit_dtype_name)
    assert callable(bicluster_sparse_input_tag)


def test_bicluster_fit_api_shell_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_fit_api_shell import (
        bicluster_fit_accept_sparse_format,
        bicluster_fit_dtype_name,
        bicluster_sparse_input_tag,
    )

    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    for estimator in (SpectralBiclustering(), SpectralCoclustering()):
        with (
            patch("sklearn.cluster._bicluster.validate_data", autospec=True, return_value=X) as mocked_validate,
            patch.object(estimator, "_check_parameters", autospec=True, return_value=None) as mocked_check,
            patch.object(estimator, "_fit", autospec=True, return_value=None) as mocked_fit,
        ):
            result = estimator.fit(X)

        assert result is estimator
        assert mocked_validate.call_args.kwargs["accept_sparse"] == "csr"
        assert mocked_validate.call_args.kwargs["dtype"] is np.float64
        mocked_check.assert_called_once()
        mocked_fit.assert_called_once_with(X)

    assert bicluster_fit_accept_sparse_format("csc") == "csr"
    assert bicluster_fit_dtype_name("numeric") == "float64"
    assert bicluster_sparse_input_tag(False) is True


def test_bicluster_fit_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_fit_api_shell import (
        bicluster_fit_accept_sparse_format,
        bicluster_fit_dtype_name,
        bicluster_sparse_input_tag,
    )

    with pytest.raises(Exception):
        bicluster_fit_accept_sparse_format((1,))  # type: ignore[arg-type]

    with pytest.raises(Exception):
        bicluster_fit_dtype_name(1)  # type: ignore[arg-type]

    with pytest.raises(Exception):
        bicluster_sparse_input_tag("false")  # type: ignore[arg-type]
