from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src/sciona/atoms/ml/sklearn/mixture/references.json"
REGISTRY_PATH = ROOT / "data/references/registry.json"
EXPECTED = {
    "gaussian_mixture_diag_fit",
    "gaussian_mixture_diag_score_samples",
    "gaussian_mixture_diag_score",
    "gaussian_mixture_diag_predict_proba",
    "gaussian_mixture_diag_predict",
    "gaussian_mixture_diag_bic",
    "gaussian_mixture_diag_aic",
    "bayesian_gaussian_mixture_diag_fit",
    "bayesian_gaussian_mixture_diag_score_samples",
    "bayesian_gaussian_mixture_diag_score",
    "bayesian_gaussian_mixture_diag_predict_proba",
    "bayesian_gaussian_mixture_diag_predict",
}


def test_gaussian_mixture_references_bind_expected_atoms_to_registry() -> None:
    refs = json.loads(REFERENCES_PATH.read_text())
    registry = json.loads(REGISTRY_PATH.read_text())
    registry_ids = set(registry["references"])
    leaf_names = {key.split("@")[0].rsplit(".", 1)[-1] for key in refs["atoms"]}
    assert leaf_names == EXPECTED
    for atom in refs["atoms"].values():
        assert atom["references"]
        for reference in atom["references"]:
            assert reference["ref_id"] in registry_ids
            metadata = reference["match_metadata"]
            assert metadata["match_type"]
            assert metadata["confidence"]
            assert metadata["notes"]


def test_gaussian_mixture_reference_line_anchors_exist() -> None:
    refs = json.loads(REFERENCES_PATH.read_text())
    for key in refs["atoms"]:
        _, anchor = key.split("@")
        file_part, line_part = anchor.rsplit(":", 1)
        source = ROOT / "src" / file_part
        assert source.exists()
        assert int(line_part) <= len(source.read_text().splitlines())
