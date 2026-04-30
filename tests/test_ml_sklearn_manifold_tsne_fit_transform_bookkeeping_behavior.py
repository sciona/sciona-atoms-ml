from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.manifold import TSNE


def test_tsne_fit_transform_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_transform_bookkeeping import (
        tsne_fit_transform_max_iter,
        tsne_fit_transform_require_single_iter_source,
        tsne_n_features_out,
        tsne_pairwise_input_tag,
    )

    assert callable(tsne_fit_transform_require_single_iter_source)
    assert callable(tsne_fit_transform_max_iter)
    assert callable(tsne_n_features_out)
    assert callable(tsne_pairwise_input_tag)


def test_tsne_fit_transform_max_iter_matches_sklearn_shell() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_transform_bookkeeping import (
        tsne_fit_transform_max_iter,
        tsne_fit_transform_require_single_iter_source,
    )

    assert tsne_fit_transform_require_single_iter_source("deprecated", None) is True
    assert tsne_fit_transform_max_iter("deprecated", None) == 1000
    assert tsne_fit_transform_max_iter("deprecated", 750) == 750
    assert tsne_fit_transform_max_iter(250, None) == 250


def test_tsne_fit_transform_conflicting_iter_args_match_sklearn_error() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_transform_bookkeeping import (
        tsne_fit_transform_max_iter,
        tsne_fit_transform_require_single_iter_source,
    )

    X = np.array([[0.0], [1.0]], dtype=float)
    with pytest.raises(ValueError, match="Both 'n_iter' and 'max_iter' attributes were set"):
        TSNE(n_iter=250, max_iter=300, perplexity=1, init="random").fit_transform(X)
    with pytest.raises(ValueError, match="Both 'n_iter' and 'max_iter' attributes were set"):
        tsne_fit_transform_require_single_iter_source(250, 300)
    with pytest.raises(ValueError, match="Both 'n_iter' and 'max_iter' attributes were set"):
        tsne_fit_transform_max_iter(250, 300)


def test_tsne_postfit_property_helpers_match_sklearn_shape_and_metric_logic() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_transform_bookkeeping import (
        tsne_n_features_out,
        tsne_pairwise_input_tag,
    )

    embedding = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64)
    assert tsne_n_features_out(embedding) == 2
    assert tsne_pairwise_input_tag("precomputed") is True
    assert tsne_pairwise_input_tag("euclidean") is False


def test_tsne_fit_transform_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_transform_bookkeeping import (
        tsne_fit_transform_max_iter,
        tsne_n_features_out,
        tsne_pairwise_input_tag,
    )

    with pytest.raises(ViolationError):
        tsne_fit_transform_max_iter("legacy", None)

    with pytest.raises(ViolationError):
        tsne_n_features_out(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        tsne_pairwise_input_tag("")
