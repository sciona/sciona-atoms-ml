from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src/sciona/atoms/ml/sklearn/neural_network/references.json"
REGISTRY_PATH = ROOT / "data/references/registry.json"

EXPECTED_ATOMS = {
    "bernoulli_rbm_fit",
    "bernoulli_rbm_free_energy",
    "bernoulli_rbm_gibbs",
    "bernoulli_rbm_mean_hiddens",
    "bernoulli_rbm_partial_fit",
    "bernoulli_rbm_sample_hiddens",
    "bernoulli_rbm_sample_visibles",
    "bernoulli_rbm_score_samples",
}


def test_bernoulli_rbm_references_cover_expected_atoms() -> None:
    refs = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]
    leaf_names = {key.split("@")[0].rsplit(".", 1)[-1] for key in refs["atoms"]}

    assert refs["schema_version"] == "1.1"
    assert leaf_names == EXPECTED_ATOMS
    for payload in refs["atoms"].values():
        assert payload["references"]
        for reference in payload["references"]:
            assert reference["ref_id"] in registry
            metadata = reference["match_metadata"]
            assert metadata["match_type"]
            assert metadata["confidence"]
            assert metadata["notes"]
