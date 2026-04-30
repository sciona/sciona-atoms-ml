from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "src/sciona/atoms/ml/tabular/gradient_boosting"


def test_tabular_gradient_boosting_references_resolve_to_registry() -> None:
    refs = json.loads((FAMILY / "references.json").read_text())
    registry = json.loads((ROOT / "data/references/registry.json").read_text())
    registry_ids = set(registry["references"])

    assert refs["schema_version"] == "1.1"
    assert refs["atoms"]

    for atom_key, entry in refs["atoms"].items():
        assert atom_key.startswith("sciona.atoms.ml.tabular.gradient_boosting.")
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            assert ref["match_metadata"]["notes"]

