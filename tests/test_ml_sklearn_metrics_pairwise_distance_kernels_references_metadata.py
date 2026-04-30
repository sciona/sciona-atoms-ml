from __future__ import annotations

import json
from pathlib import Path


REFERENCES_PATH = Path(
    "src/sciona/atoms/ml/sklearn/metrics_pairwise_distance_kernels/references.json"
)

EXPECTED = {
    "sciona.atoms.ml.sklearn.metrics_pairwise_distance_kernels.pairwise_rbf_kernel",
    "sciona.atoms.ml.sklearn.metrics_pairwise_distance_kernels.pairwise_additive_chi2_kernel",
    "sciona.atoms.ml.sklearn.metrics_pairwise_distance_kernels.pairwise_chi2_kernel",
}


def test_pairwise_distance_kernel_references_cover_atoms() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    found = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert found == EXPECTED


def test_pairwise_distance_kernel_references_use_known_sources() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for atom in payload["atoms"].values():
        ref_ids = {reference["ref_id"] for reference in atom["references"]}
        assert "repo_sklearn" in ref_ids
        assert "pedregosa2011sklearn" in ref_ids
