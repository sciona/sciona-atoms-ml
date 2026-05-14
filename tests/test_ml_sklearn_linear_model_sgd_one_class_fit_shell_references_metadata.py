from __future__ import annotations

import importlib
import json
from pathlib import Path


FAMILY = "sgd_one_class_fit_shell"
MODULE = f"sciona.atoms.ml.sklearn.linear_model.{FAMILY}"
FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/linear_model") / FAMILY

EXPECTED_ATOMS = {
    f"{MODULE}.sgd_one_class_target",
    f"{MODULE}.sgd_one_class_fixed_solver_context",
    f"{MODULE}.sgd_one_class_validation_sample_mask",
    f"{MODULE}.sgd_one_class_intercept_from_offset",
    f"{MODULE}.sgd_one_class_offset_from_intercept",
    f"{MODULE}.sgd_one_class_time_step_after_fit",
    f"{MODULE}.sgd_one_class_average_active",
    f"{MODULE}.sgd_one_class_average_buffers",
    f"{MODULE}.sgd_one_class_parameter_allocation_payload",
    f"{MODULE}.sgd_one_class_fit_one_class_payload",
    f"{MODULE}.sgd_one_class_partial_fit_result",
}


def test_sgd_one_class_fit_shell_references_cover_registered_atoms() -> None:
    module = importlib.import_module(MODULE)
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    assert references["schema_version"] == "1.1"
    assert set(module.__all__) == {
        "sgd_one_class_target",
        "sgd_one_class_fixed_solver_context",
        "sgd_one_class_validation_sample_mask",
        "sgd_one_class_intercept_from_offset",
        "sgd_one_class_offset_from_intercept",
        "sgd_one_class_time_step_after_fit",
        "sgd_one_class_average_active",
        "sgd_one_class_average_buffers",
        "sgd_one_class_parameter_allocation_payload",
        "sgd_one_class_fit_one_class_payload",
        "sgd_one_class_partial_fit_result",
    }

    referenced_atoms = {key.split("@", 1)[0] for key in references["atoms"]}
    assert referenced_atoms == EXPECTED_ATOMS


def test_sgd_one_class_fit_shell_references_have_source_and_project_refs() -> None:
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    for atom_key, payload in references["atoms"].items():
        assert atom_key.startswith(MODULE)
        ref_ids = {entry["ref_id"] for entry in payload["references"]}
        assert {"repo_sklearn", "pedregosa2011sklearn"} <= ref_ids
        for entry in payload["references"]:
            assert entry["match_metadata"]["confidence"] in {"high", "medium"}
            assert entry["match_metadata"]["match_type"].startswith("manual_")
