from __future__ import annotations

import importlib
import json
from pathlib import Path


FAMILY = "logistic_scoring_path_callback_shell"
MODULE = f"sciona.atoms.ml.sklearn.linear_model.{FAMILY}"
FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/linear_model") / FAMILY

EXPECTED_ATOMS = {
    f"{MODULE}.logistic_scoring_classes",
    f"{MODULE}.logistic_scoring_coef_intercept_state",
    f"{MODULE}.logistic_scoring_fold_split",
    f"{MODULE}.logistic_scoring_path_call",
    f"{MODULE}.logistic_scoring_path_kwargs",
    f"{MODULE}.logistic_scoring_positive_y_test",
    f"{MODULE}.logistic_scoring_result_tuple",
    f"{MODULE}.logistic_scoring_sample_weight_split",
    f"{MODULE}.logistic_scoring_score_call_payload",
    f"{MODULE}.logistic_scoring_score_params",
    f"{MODULE}.logistic_scoring_temp_log_reg_kwargs",
}


def test_logistic_scoring_path_callback_shell_references_cover_registered_atoms() -> None:
    module = importlib.import_module(MODULE)
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    assert references["schema_version"] == "1.1"
    assert set(module.__all__) == {atom.rsplit(".", 1)[-1] for atom in EXPECTED_ATOMS}

    referenced_atoms = {key.split("@", 1)[0] for key in references["atoms"]}
    assert referenced_atoms == EXPECTED_ATOMS


def test_logistic_scoring_path_callback_shell_references_have_source_and_project_refs() -> None:
    references = json.loads((FAMILY_DIR / "references.json").read_text())

    for atom_key, payload in references["atoms"].items():
        assert atom_key.startswith(MODULE)
        ref_ids = {entry["ref_id"] for entry in payload["references"]}
        assert {"repo_sklearn", "pedregosa2011sklearn"} <= ref_ids
        for entry in payload["references"]:
            assert entry["match_metadata"]["confidence"] in {"high", "medium"}
            assert entry["match_metadata"]["match_type"].startswith("manual_")


def test_logistic_scoring_path_callback_shell_cdg_atomic_nodes_have_complexity_metadata() -> None:
    cdg = json.loads((FAMILY_DIR / "cdg.json").read_text())
    atomic_nodes = [node for node in cdg["nodes"] if node.get("status") == "atomic"]

    assert {node["name"] for node in atomic_nodes} == {atom.rsplit(".", 1)[-1] for atom in EXPECTED_ATOMS}
    for node in atomic_nodes:
        assert isinstance(node["time_complexity"], str) and node["time_complexity"].startswith("O(")
        assert isinstance(node["space_complexity"], str) and node["space_complexity"].startswith("O(")
        assert isinstance(node["complexity_reasoning"], str) and node["complexity_reasoning"]
        assert isinstance(node["complexity_confidence"], int)
        assert 1 <= node["complexity_confidence"] <= 100
