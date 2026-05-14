from __future__ import annotations

import importlib
import json
from pathlib import Path


FAMILY = "huber_fit_optimizer_shell"
MODULE = f"sciona.atoms.ml.sklearn.linear_model.{FAMILY}"
FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/linear_model") / FAMILY

EXPECTED_ATOMS = {
    f"{MODULE}.huber_fit_initial_parameters",
    f"{MODULE}.huber_fit_bounds",
    f"{MODULE}.huber_fit_optimizer_payload",
    f"{MODULE}.huber_fit_status2_failure_message",
    f"{MODULE}.huber_fit_result_attributes",
    f"{MODULE}.huber_fit_outlier_handoff_payload",
}


def test_huber_fit_optimizer_shell_references_cover_registered_atoms() -> None:
    module = importlib.import_module(MODULE)
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    assert references["schema_version"] == "1.1"
    assert set(module.__all__) == {
        "huber_fit_initial_parameters",
        "huber_fit_bounds",
        "huber_fit_optimizer_payload",
        "huber_fit_status2_failure_message",
        "huber_fit_result_attributes",
        "huber_fit_outlier_handoff_payload",
    }

    referenced_atoms = {key.split("@", 1)[0] for key in references["atoms"]}
    assert referenced_atoms == EXPECTED_ATOMS


def test_huber_fit_optimizer_shell_references_have_source_and_project_refs() -> None:
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    for atom_key, payload in references["atoms"].items():
        assert atom_key.startswith(MODULE)
        ref_ids = {entry["ref_id"] for entry in payload["references"]}
        assert {"repo_sklearn", "pedregosa2011sklearn"} <= ref_ids
        for entry in payload["references"]:
            assert entry["match_metadata"]["confidence"] in {"high", "medium"}
            assert entry["match_metadata"]["match_type"].startswith("manual_")
