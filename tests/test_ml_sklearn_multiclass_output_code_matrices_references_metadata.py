from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "multiclass" / "output_code_matrices" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.multiclass.output_code_matrices.output_code_fit_classes",
    "sciona.atoms.ml.sklearn.multiclass.output_code_matrices.output_code_target_matrix",
    "sciona.atoms.ml.sklearn.multiclass.output_code_matrices.output_code_response_matrix",
}


def test_output_code_matrices_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_output_code_matrices_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]
    for atom_entry in payload["atoms"].values():
        refs = atom_entry["references"]
        assert refs
        for ref in refs:
            ref_id = ref["ref_id"]
            assert ref_id in registry
            assert ref["match_metadata"]["confidence"] == "high"


def test_output_code_matrices_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.multiclass.output_code_matrices.atoms")
    for fqdn in EXPECTED_FQDNS:
        leaf = fqdn.rsplit(".", 1)[-1]
        assert leaf in {"output_code_fit_classes", "output_code_target_matrix", "output_code_response_matrix"}
