from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "inspection" / "partial_dependence_task_guards" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_task_guards.partial_dependence_require_classifier_or_regressor",
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_task_guards.partial_dependence_require_not_multiclass_multioutput",
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_task_guards.partial_dependence_resolve_recursion_response_method",
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_task_guards.partial_dependence_require_decision_function_for_recursion",
}


def test_partial_dependence_task_guards_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_partial_dependence_task_guards_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]
    for atom_entry in payload["atoms"].values():
        refs = atom_entry["references"]
        assert refs
        for ref in refs:
            ref_id = ref["ref_id"]
            assert ref_id in registry
            assert ref["match_metadata"]["confidence"] == "high"


def test_partial_dependence_task_guards_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.inspection.partial_dependence_task_guards.atoms")
    observed = {fqdn.rsplit(".", 1)[-1] for fqdn in EXPECTED_FQDNS}
    assert observed == {
        "partial_dependence_require_classifier_or_regressor",
        "partial_dependence_require_not_multiclass_multioutput",
        "partial_dependence_resolve_recursion_response_method",
        "partial_dependence_require_decision_function_for_recursion",
    }
