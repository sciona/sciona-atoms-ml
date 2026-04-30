from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.decomposition import LatentDirichletAllocation


def _tiny_counts() -> np.ndarray:
    return np.array(
        [
            [1.0, 2.0, 0.0, 1.0],
            [0.0, 1.0, 3.0, 0.0],
            [2.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_lda_postfit_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_postfit_shell import (
        lda_n_features_out,
        lda_score_from_bound,
        lda_transform_output,
        lda_unnormalized_transform_output,
    )

    assert callable(lda_unnormalized_transform_output)
    assert callable(lda_transform_output)
    assert callable(lda_score_from_bound)
    assert callable(lda_n_features_out)


def test_lda_transform_and_score_shell_helpers_match_sklearn_behavior() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_postfit_shell import (
        lda_n_features_out,
        lda_score_from_bound,
        lda_transform_output,
        lda_unnormalized_transform_output,
    )

    X = _tiny_counts()
    model = LatentDirichletAllocation(n_components=2, random_state=0, max_iter=1)
    model.fit(X)

    doc_topic = model._unnormalized_transform(X)
    assert np.allclose(lda_unnormalized_transform_output(doc_topic), doc_topic)
    assert np.allclose(lda_transform_output(doc_topic, normalize=False), doc_topic)
    assert np.allclose(lda_transform_output(doc_topic, normalize=True), model.transform(X, normalize=True))

    score = model._approx_bound(X, doc_topic, sub_sampling=False)
    assert lda_score_from_bound(score) == pytest.approx(model.score(X))
    assert lda_n_features_out(model.n_components) == model._n_features_out


def test_lda_postfit_shell_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_postfit_shell import (
        lda_n_features_out,
        lda_score_from_bound,
        lda_transform_output,
        lda_unnormalized_transform_output,
    )

    with pytest.raises(ViolationError):
        lda_unnormalized_transform_output(np.array([[1.0, 0.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        lda_transform_output(np.array([[0.0, 1.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        lda_score_from_bound(float("nan"))

    with pytest.raises(ViolationError):
        lda_n_features_out(0)
