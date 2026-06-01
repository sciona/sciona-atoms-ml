from __future__ import annotations

import importlib
import json
from pathlib import Path


FAMILY = "ransac_metadata_routing_shell"
MODULE = f"sciona.atoms.ml.sklearn.linear_model.{FAMILY}"
FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/linear_model") / FAMILY

EXPECTED_ATOMS = {
    f"{MODULE}.ransac_metadata_router_owner",
    f"{MODULE}.ransac_metadata_method_mapping_add_kwargs",
    f"{MODULE}.ransac_metadata_estimator_payload",
    f"{MODULE}.ransac_metadata_router_result",
}


def test_ransac_metadata_routing_shell_references_cover_registered_atoms() -> None:
    module = importlib.import_module(MODULE)
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    assert references["schema_version"] == "1.1"
    assert set(module.__all__) == {
        "ransac_metadata_router_owner",
        "ransac_metadata_method_mapping_add_kwargs",
        "ransac_metadata_estimator_payload",
        "ransac_metadata_router_result",
    }

    referenced_atoms = {key.split("@", 1)[0] for key in references["atoms"]}
    assert referenced_atoms == EXPECTED_ATOMS


def test_ransac_metadata_routing_shell_references_have_source_and_project_refs() -> None:
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    for atom_key, payload in references["atoms"].items():
        assert atom_key.startswith(MODULE)
        ref_ids = {entry["ref_id"] for entry in payload["references"]}
        assert {"repo_sklearn", "pedregosa2011sklearn"} <= ref_ids
        for entry in payload["references"]:
            assert entry["match_metadata"]["confidence"] in {"high", "medium"}
            assert entry["match_metadata"]["match_type"].startswith("manual_")
