from __future__ import annotations

import importlib
import json
from pathlib import Path


FAMILY = "logistic_cv_best_refit_selection_shell"
MODULE = f"sciona.atoms.ml.sklearn.linear_model.{FAMILY}"
FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/linear_model") / FAMILY

EXPECTED_ATOMS = {
    f"{MODULE}.logistic_cv_loop_path_views",
    f"{MODULE}.logistic_cv_best_flat_index",
    f"{MODULE}.logistic_cv_best_C_l1_selection",
    f"{MODULE}.logistic_cv_refit_coef_init",
    f"{MODULE}.logistic_cv_nonrefit_best_indices",
    f"{MODULE}.logistic_cv_nonrefit_average_w",
    f"{MODULE}.logistic_cv_nonrefit_average_C",
    f"{MODULE}.logistic_cv_nonrefit_average_l1_ratio",
    f"{MODULE}.logistic_cv_multinomial_final_components",
    f"{MODULE}.logistic_cv_ovr_final_row",
}


def test_logistic_cv_best_refit_selection_shell_references_cover_registered_atoms() -> None:
    module = importlib.import_module(MODULE)
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    assert references["schema_version"] == "1.1"
    assert set(module.__all__) == {
        "logistic_cv_loop_path_views",
        "logistic_cv_best_flat_index",
        "logistic_cv_best_C_l1_selection",
        "logistic_cv_refit_coef_init",
        "logistic_cv_nonrefit_best_indices",
        "logistic_cv_nonrefit_average_w",
        "logistic_cv_nonrefit_average_C",
        "logistic_cv_nonrefit_average_l1_ratio",
        "logistic_cv_multinomial_final_components",
        "logistic_cv_ovr_final_row",
    }

    referenced_atoms = {key.split("@", 1)[0] for key in references["atoms"]}
    assert referenced_atoms == EXPECTED_ATOMS


def test_logistic_cv_best_refit_selection_shell_references_have_source_and_project_refs() -> None:
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    for atom_key, payload in references["atoms"].items():
        assert atom_key.startswith(MODULE)
        ref_ids = {entry["ref_id"] for entry in payload["references"]}
        assert {"repo_sklearn", "pedregosa2011sklearn"} <= ref_ids
        for entry in payload["references"]:
            assert entry["match_metadata"]["confidence"] in {"high", "medium"}
            assert entry["match_metadata"]["match_type"].startswith("manual_")
