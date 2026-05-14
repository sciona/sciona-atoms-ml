from __future__ import annotations

import importlib
import json
from pathlib import Path


FAMILY = "huber_tags_super_callback_shell"
MODULE = f"sciona.atoms.ml.sklearn.linear_model.{FAMILY}"
FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/linear_model") / FAMILY

EXPECTED_ATOMS = {
    f"{MODULE}.huber_tags_super_result",
    f"{MODULE}.huber_tags_sparse_input_value",
    f"{MODULE}.huber_tags_return",
}


def test_huber_tags_super_callback_shell_references_cover_registered_atoms() -> None:
    module = importlib.import_module(MODULE)
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    assert references["schema_version"] == "1.1"
    assert set(module.__all__) == {
        "huber_tags_super_result",
        "huber_tags_sparse_input_value",
        "huber_tags_return",
    }

    referenced_atoms = {key.split("@", 1)[0] for key in references["atoms"]}
    assert referenced_atoms == EXPECTED_ATOMS


def test_huber_tags_super_callback_shell_references_have_source_and_project_refs() -> None:
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    for atom_key, payload in references["atoms"].items():
        assert atom_key.startswith(MODULE)
        ref_ids = {entry["ref_id"] for entry in payload["references"]}
        assert {"repo_sklearn", "pedregosa2011sklearn"} <= ref_ids
        for entry in payload["references"]:
            assert entry["match_metadata"]["confidence"] in {"high", "medium"}
            assert entry["match_metadata"]["match_type"].startswith("manual_")
