from __future__ import annotations

import importlib
import json
from pathlib import Path


FAMILY = "quantile_solver_guard_shell"
MODULE = f"sciona.atoms.ml.sklearn.linear_model.{FAMILY}"
FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/linear_model") / FAMILY

EXPECTED_ATOMS = {
    f"{MODULE}.quantile_interior_point_removed_guard",
    f"{MODULE}.quantile_interior_point_removed_message",
    f"{MODULE}.quantile_sparse_solver_guard",
    f"{MODULE}.quantile_sparse_solver_message",
    f"{MODULE}.quantile_solver_options_payload",
}


def test_quantile_solver_guard_shell_references_cover_registered_atoms() -> None:
    module = importlib.import_module(MODULE)
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    assert references["schema_version"] == "1.1"
    assert set(module.__all__) == {
        "quantile_interior_point_removed_guard",
        "quantile_interior_point_removed_message",
        "quantile_sparse_solver_guard",
        "quantile_sparse_solver_message",
        "quantile_solver_options_payload",
    }

    referenced_atoms = {key.split("@", 1)[0] for key in references["atoms"]}
    assert referenced_atoms == EXPECTED_ATOMS


def test_quantile_solver_guard_shell_references_have_source_and_project_refs() -> None:
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    for atom_key, payload in references["atoms"].items():
        assert atom_key.startswith(MODULE)
        ref_ids = {entry["ref_id"] for entry in payload["references"]}
        assert {"repo_sklearn", "pedregosa2011sklearn"} <= ref_ids
        for entry in payload["references"]:
            assert entry["match_metadata"]["confidence"] in {"high", "medium"}
            assert entry["match_metadata"]["match_type"].startswith("manual_")
