from __future__ import annotations

import importlib
import json
from pathlib import Path


FAMILY = "glm_score_deviance_tail"
MODULE = f"sciona.atoms.ml.sklearn.linear_model.{FAMILY}"
FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/linear_model") / FAMILY

EXPECTED_ATOMS = {
    f"{MODULE}.glm_score_constant_average",
    f"{MODULE}.glm_score_d2_from_deviances",
    f"{MODULE}.glm_score_null_raw_prediction",
    f"{MODULE}.glm_score_sample_weight_check_args",
    f"{MODULE}.glm_score_sample_weight_check_kwargs",
    f"{MODULE}.glm_score_target_range_error_message",
    f"{MODULE}.glm_score_y_check_array_kwargs",
}


def test_glm_score_deviance_tail_references_cover_registered_atoms() -> None:
    module = importlib.import_module(MODULE)
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    assert references["schema_version"] == "1.1"
    assert set(module.__all__) == {
        "glm_score_constant_average",
        "glm_score_d2_from_deviances",
        "glm_score_null_raw_prediction",
        "glm_score_sample_weight_check_args",
        "glm_score_sample_weight_check_kwargs",
        "glm_score_target_range_error_message",
        "glm_score_y_check_array_kwargs",
    }

    referenced_atoms = {key.split("@", 1)[0] for key in references["atoms"]}
    assert referenced_atoms == EXPECTED_ATOMS


def test_glm_score_deviance_tail_references_have_source_and_project_refs() -> None:
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    for atom_key, payload in references["atoms"].items():
        assert atom_key.startswith(MODULE)
        ref_ids = {entry["ref_id"] for entry in payload["references"]}
        assert {"repo_sklearn", "pedregosa2011sklearn"} <= ref_ids
        for entry in payload["references"]:
            metadata = entry["match_metadata"]
            assert metadata["confidence"] in {"high", "medium"}
            assert metadata["match_type"].startswith("manual_")
