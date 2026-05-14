from __future__ import annotations

import importlib
import json
from pathlib import Path


FAMILY = "ransac_callback_orchestration_shell"
MODULE = f"sciona.atoms.ml.sklearn.linear_model.{FAMILY}"
FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/linear_model") / FAMILY

EXPECTED_ATOMS = {
    f"{MODULE}.ransac_nonrouting_estimator_params",
    f"{MODULE}.ransac_skip_limit_exceeded",
    f"{MODULE}.ransac_trial_subset_payload",
    f"{MODULE}.ransac_estimator_fit_callback_payload",
    f"{MODULE}.ransac_inlier_subset_payload",
    f"{MODULE}.ransac_score_callback_payload",
    f"{MODULE}.ransac_no_consensus_failure_message",
    f"{MODULE}.ransac_final_refit_callback_payload",
}


def test_ransac_callback_orchestration_shell_references_cover_registered_atoms() -> None:
    module = importlib.import_module(MODULE)
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    assert references["schema_version"] == "1.1"
    assert set(module.__all__) == {
        "ransac_nonrouting_estimator_params",
        "ransac_skip_limit_exceeded",
        "ransac_trial_subset_payload",
        "ransac_estimator_fit_callback_payload",
        "ransac_inlier_subset_payload",
        "ransac_score_callback_payload",
        "ransac_no_consensus_failure_message",
        "ransac_final_refit_callback_payload",
    }

    referenced_atoms = {key.split("@", 1)[0] for key in references["atoms"]}
    assert referenced_atoms == EXPECTED_ATOMS


def test_ransac_callback_orchestration_shell_references_have_source_and_project_refs() -> None:
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    for atom_key, payload in references["atoms"].items():
        assert atom_key.startswith(MODULE)
        ref_ids = {entry["ref_id"] for entry in payload["references"]}
        assert {"repo_sklearn", "pedregosa2011sklearn"} <= ref_ids
        for entry in payload["references"]:
            assert entry["match_metadata"]["confidence"] in {"high", "medium"}
            assert entry["match_metadata"]["match_type"].startswith("manual_")
