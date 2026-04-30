from __future__ import annotations

from types import MethodType

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


def test_lda_em_updates_imports() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_em_updates import (
        lda_batch_components,
        lda_e_step_document_topic_matrix,
        lda_e_step_sufficient_statistics,
        lda_online_components,
        lda_online_document_ratio,
        lda_online_update_weight,
    )

    assert callable(lda_e_step_document_topic_matrix)
    assert callable(lda_e_step_sufficient_statistics)
    assert callable(lda_online_update_weight)
    assert callable(lda_online_document_ratio)
    assert callable(lda_batch_components)
    assert callable(lda_online_components)


def test_lda_e_step_merge_helpers_match_private_merge_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_em_updates import (
        lda_e_step_document_topic_matrix,
        lda_e_step_sufficient_statistics,
    )

    doc_topics = (
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([[5.0, 6.0]], dtype=np.float64),
    )
    assert np.array_equal(
        lda_e_step_document_topic_matrix(doc_topics),
        np.vstack(doc_topics),
    )

    sstats_blocks = (
        np.array([[1.0, 0.5], [0.25, 2.0]], dtype=np.float64),
        np.array([[0.5, 1.5], [1.0, 0.5]], dtype=np.float64),
    )
    exp_dirichlet_component = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    expected = np.zeros(exp_dirichlet_component.shape, dtype=np.float64)
    for block in sstats_blocks:
        expected += block
    expected *= exp_dirichlet_component

    observed = lda_e_step_sufficient_statistics(sstats_blocks, exp_dirichlet_component)
    assert np.allclose(observed, expected)


def test_lda_component_update_helpers_match_private_em_step_branches() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_em_updates import (
        lda_batch_components,
        lda_online_components,
        lda_online_document_ratio,
        lda_online_update_weight,
    )

    X = _tiny_counts()
    suff_stats = np.array(
        [
            [0.4, 0.8, 1.2, 0.6],
            [0.3, 0.5, 0.7, 0.9],
        ],
        dtype=np.float64,
    )

    batch_model = LatentDirichletAllocation(n_components=2, random_state=0)
    batch_model._init_latent_vars(X.shape[1], dtype=X.dtype)
    batch_model._e_step = MethodType(lambda self, X, cal_sstats, random_init, parallel=None: (None, suff_stats), batch_model)
    batch_model._em_step(X, total_samples=X.shape[0], batch_update=True)

    expected_batch = lda_batch_components(batch_model.topic_word_prior_, suff_stats)
    assert np.allclose(batch_model.components_, expected_batch)

    online_model = LatentDirichletAllocation(
        n_components=2,
        random_state=0,
        learning_offset=10.0,
        learning_decay=0.7,
    )
    online_model._init_latent_vars(X.shape[1], dtype=X.dtype)
    previous_components = online_model.components_.copy()
    n_batch_iter = online_model.n_batch_iter_
    online_model._e_step = MethodType(lambda self, X, cal_sstats, random_init, parallel=None: (None, suff_stats), online_model)
    online_model._em_step(X, total_samples=9, batch_update=False)

    weight = lda_online_update_weight(online_model.learning_offset, n_batch_iter, online_model.learning_decay)
    doc_ratio = lda_online_document_ratio(9.0, X.shape[0])
    expected_online = lda_online_components(
        previous_components,
        online_model.topic_word_prior_,
        suff_stats,
        weight=weight,
        doc_ratio=doc_ratio,
    )
    assert np.allclose(online_model.components_, expected_online)


def test_lda_em_updates_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_em_updates import (
        lda_batch_components,
        lda_e_step_document_topic_matrix,
        lda_e_step_sufficient_statistics,
        lda_online_components,
        lda_online_document_ratio,
        lda_online_update_weight,
    )

    with pytest.raises(ViolationError):
        lda_e_step_document_topic_matrix((np.array([[1.0, 2.0]]), np.array([[3.0, 4.0, 5.0]])))

    with pytest.raises(ViolationError):
        lda_e_step_sufficient_statistics((np.array([[1.0, -1.0]]),), np.array([[1.0, 2.0]]))

    with pytest.raises(ViolationError):
        lda_online_update_weight(10.0, 0, 0.7)

    with pytest.raises(ViolationError):
        lda_online_document_ratio(5.0, 0)

    with pytest.raises(ViolationError):
        lda_batch_components(0.5, np.array([[1.0, np.nan]]))

    with pytest.raises(ViolationError):
        lda_online_components(
            np.array([[1.0, 2.0]]),
            0.5,
            np.array([[1.0, 2.0]]),
            weight=1.2,
            doc_ratio=1.0,
        )
