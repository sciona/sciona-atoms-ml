from __future__ import annotations

import importlib
import json
from pathlib import Path


FAMILY = "lars_cv_orchestration_shell"
MODULE = f"sciona.atoms.ml.sklearn.linear_model.{FAMILY}"
FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/linear_model") / FAMILY

EXPECTED_ATOMS = {
    f"{MODULE}.lars_cv_path_residues_callback_kwargs",
}


def test_lars_cv_orchestration_shell_references_cover_registered_atoms() -> None:
    module = importlib.import_module(MODULE)
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    assert references["schema_version"] == "1.1"
    assert set(module.__all__) == {"lars_cv_path_residues_callback_kwargs"}

    referenced_atoms = {key.split("@", 1)[0] for key in references["atoms"]}
    assert referenced_atoms == EXPECTED_ATOMS


def test_lars_cv_orchestration_shell_references_have_source_and_project_refs() -> None:
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    for atom_key, payload in references["atoms"].items():
        assert atom_key.startswith(MODULE)
        ref_ids = {entry["ref_id"] for entry in payload["references"]}
        assert {"repo_sklearn", "pedregosa2011sklearn"} <= ref_ids
        for entry in payload["references"]:
            metadata = entry["match_metadata"]
            assert metadata["confidence"] in {"high", "medium"}
            assert metadata["match_type"].startswith("manual_")
