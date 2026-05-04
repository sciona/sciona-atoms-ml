from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.cluster import SpectralClustering


def test_spectral_fit_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_fit_api_shell import (
        spectral_fit_accept_sparse_formats,
        spectral_fit_affinity_allows_square_input,
        spectral_fit_dtype_name,
        spectral_fit_square_input_warning_required,
        spectral_pairwise_input_tag,
    )

    assert callable(spectral_fit_accept_sparse_formats)
    assert callable(spectral_fit_affinity_allows_square_input)
    assert callable(spectral_fit_dtype_name)
    assert callable(spectral_fit_square_input_warning_required)
    assert callable(spectral_pairwise_input_tag)


def test_spectral_fit_api_shell_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_fit_api_shell import (
        spectral_fit_accept_sparse_formats,
        spectral_fit_affinity_allows_square_input,
        spectral_fit_dtype_name,
        spectral_fit_square_input_warning_required,
        spectral_pairwise_input_tag,
    )

    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    estimator = SpectralClustering(affinity="rbf")

    with (
        patch("sklearn.cluster._spectral.validate_data", autospec=True, return_value=X) as mocked_validate,
        patch("sklearn.cluster._spectral.pairwise_kernels", autospec=True, return_value=np.eye(2, dtype=np.float64)),
        patch("sklearn.cluster._spectral._spectral_embedding", autospec=True, return_value=np.eye(2, dtype=np.float64)),
        patch("sklearn.cluster._spectral.k_means", autospec=True, return_value=(None, np.array([0, 1]), None)),
        patch("sklearn.cluster._spectral.warnings.warn", autospec=True) as mocked_warn,
    ):
        result = estimator.fit(X)

    assert result is estimator
    assert mocked_validate.call_args.kwargs["accept_sparse"] == ["csr", "csc", "coo"]
    assert mocked_validate.call_args.kwargs["dtype"] is np.float64
    assert mocked_validate.call_args.kwargs["ensure_min_samples"] == 2
    mocked_warn.assert_called_once()

    assert spectral_fit_accept_sparse_formats(("csr",)) == ("csr", "csc", "coo")
    assert spectral_fit_dtype_name("numeric") == "float64"
    assert spectral_fit_affinity_allows_square_input("precomputed") is True
    assert spectral_fit_affinity_allows_square_input("rbf") is False
    assert spectral_fit_square_input_warning_required("rbf", (2, 2)) is True
    assert spectral_fit_square_input_warning_required("precomputed", (2, 2)) is False
    assert spectral_pairwise_input_tag("precomputed_nearest_neighbors", False) is True
    assert spectral_pairwise_input_tag("nearest_neighbors", False) is False


def test_spectral_fit_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_fit_api_shell import (
        spectral_fit_accept_sparse_formats,
        spectral_fit_dtype_name,
        spectral_fit_square_input_warning_required,
        spectral_pairwise_input_tag,
    )

    with pytest.raises(Exception):
        spectral_fit_accept_sparse_formats(("csr", 1))  # type: ignore[arg-type]

    with pytest.raises(Exception):
        spectral_fit_dtype_name(1)  # type: ignore[arg-type]

    with pytest.raises(Exception):
        spectral_fit_square_input_warning_required("", (2, 2))

    with pytest.raises(Exception):
        spectral_pairwise_input_tag("rbf", "false")  # type: ignore[arg-type]
