from __future__ import annotations

import importlib
import json
from pathlib import Path


FAMILY = "glm_tags_loss_callback_shell"
MODULE = f"sciona.atoms.ml.sklearn.linear_model.{FAMILY}"
FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/linear_model") / FAMILY

EXPECTED_ATOMS = {
    f"{MODULE}.glm_tags_super_result",
    f"{MODULE}.glm_tags_sparse_input_value",
    f"{MODULE}.glm_tags_loss_callback_result",
    f"{MODULE}.glm_tags_positive_only_from_negative_range",
    f"{MODULE}.glm_tags_exception_fallback",
    f"{MODULE}.glm_tags_return",
    f"{MODULE}.glm_base_default_loss_name",
}


def test_glm_tags_loss_callback_shell_references_cover_registered_atoms() -> None:
    module = importlib.import_module(MODULE)
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    assert references["schema_version"] == "1.1"
    assert set(module.__all__) == {
        "glm_tags_super_result",
        "glm_tags_sparse_input_value",
        "glm_tags_loss_callback_result",
        "glm_tags_positive_only_from_negative_range",
        "glm_tags_exception_fallback",
        "glm_tags_return",
        "glm_base_default_loss_name",
    }

    referenced_atoms = {key.split("@", 1)[0] for key in references["atoms"]}
    assert referenced_atoms == EXPECTED_ATOMS


def test_glm_tags_loss_callback_shell_references_have_source_and_project_refs() -> None:
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    for atom_key, payload in references["atoms"].items():
        assert atom_key.startswith(MODULE)
        ref_ids = {entry["ref_id"] for entry in payload["references"]}
        assert {"repo_sklearn", "pedregosa2011sklearn"} <= ref_ids
        for entry in payload["references"]:
            assert entry["match_metadata"]["confidence"] in {"high", "medium"}
            assert entry["match_metadata"]["match_type"].startswith("manual_")
