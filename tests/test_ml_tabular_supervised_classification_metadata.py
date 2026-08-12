from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "src/sciona/atoms/ml/tabular/supervised_classification"
PREFIX = "sciona.atoms.ml.tabular.supervised_classification."


def test_supervised_tabular_references_resolve_to_registry() -> None:
    references = json.loads((FAMILY / "references.json").read_text())
    registry = json.loads((ROOT / "data/references/registry.json").read_text())
    registry_ids = set(registry["references"])
    assert len(references["atoms"]) == 6
    for key, entry in references["atoms"].items():
        assert key.startswith(PREFIX)
        assert entry["references"]
        assert all(reference["ref_id"] in registry_ids for reference in entry["references"])


def test_supervised_tabular_review_bundle_covers_family() -> None:
    bundle = json.loads(
        (ROOT / "data/review_bundles/ml_tabular_supervised_classification.review_bundle.json").read_text()
    )
    rows = {row["atom_name"] for row in bundle["rows"]}
    expected = {
        PREFIX + "stratified_tabular_split",
        PREFIX + "fit_prior_probability",
        PREFIX + "predict_prior_probabilities",
        PREFIX + "fit_one_hot_logistic",
        PREFIX + "fit_cross_validated_logistic",
        PREFIX + "predict_binary_probabilities",
    }
    assert rows == expected
    assert bundle["review_status"] == "reviewed"
    assert not bundle["blocking_findings"]


def test_supervised_tabular_cdg_has_grounded_contracts() -> None:
    cdg = json.loads((FAMILY / "cdg.json").read_text())
    leaves = [node for node in cdg["nodes"] if node["status"] == "atomic"]
    assert len(leaves) == 6
    assert all(node["matched_primitive"].startswith(PREFIX) for node in leaves)
    assert all(node["inputs"] and node["outputs"] for node in leaves)
