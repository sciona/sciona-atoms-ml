from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "preprocessing" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.preprocessing.add_dummy_feature",
    "sciona.atoms.ml.sklearn.preprocessing.binarize",
    "sciona.atoms.ml.sklearn.preprocessing.binarizer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.kernel_centerer_fit",
    "sciona.atoms.ml.sklearn.preprocessing.kernel_centerer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.label_binarize",
    "sciona.atoms.ml.sklearn.preprocessing.label_binarizer_fit",
    "sciona.atoms.ml.sklearn.preprocessing.label_binarizer_fit_transform",
    "sciona.atoms.ml.sklearn.preprocessing.label_binarizer_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.label_binarizer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.label_encoder_fit",
    "sciona.atoms.ml.sklearn.preprocessing.label_encoder_fit_transform",
    "sciona.atoms.ml.sklearn.preprocessing.label_encoder_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.label_encoder_transform",
    "sciona.atoms.ml.sklearn.preprocessing.maxabs_scale",
    "sciona.atoms.ml.sklearn.preprocessing.maxabs_scaler_fit",
    "sciona.atoms.ml.sklearn.preprocessing.maxabs_scaler_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.maxabs_scaler_partial_fit",
    "sciona.atoms.ml.sklearn.preprocessing.maxabs_scaler_transform",
    "sciona.atoms.ml.sklearn.preprocessing.minmax_scale",
    "sciona.atoms.ml.sklearn.preprocessing.minmax_scaler_fit",
    "sciona.atoms.ml.sklearn.preprocessing.minmax_scaler_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.minmax_scaler_partial_fit",
    "sciona.atoms.ml.sklearn.preprocessing.minmax_scaler_transform",
    "sciona.atoms.ml.sklearn.preprocessing.multi_label_binarizer_fit",
    "sciona.atoms.ml.sklearn.preprocessing.multi_label_binarizer_fit_transform",
    "sciona.atoms.ml.sklearn.preprocessing.multi_label_binarizer_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.multi_label_binarizer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.normalize",
    "sciona.atoms.ml.sklearn.preprocessing.normalizer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.robust_scale",
    "sciona.atoms.ml.sklearn.preprocessing.robust_scaler_fit",
    "sciona.atoms.ml.sklearn.preprocessing.robust_scaler_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.robust_scaler_transform",
    "sciona.atoms.ml.sklearn.preprocessing.scale",
    "sciona.atoms.ml.sklearn.preprocessing.standard_scaler_fit",
    "sciona.atoms.ml.sklearn.preprocessing.standard_scaler_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.standard_scaler_partial_fit",
    "sciona.atoms.ml.sklearn.preprocessing.standard_scaler_transform",
}


def test_references_json_exists_and_has_all_preprocessing_fqdns() -> None:
    assert REFERENCES_PATH.exists()
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_keys = set(payload["atoms"])
    assert len(atom_keys) == 39
    assert {key.partition("@")[0] for key in atom_keys} == EXPECTED_FQDNS


def test_each_atom_has_nonempty_references() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        assert entry["references"], f"empty references for {key}"


def test_ref_ids_exist_in_local_registry() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])
    for key, entry in payload["atoms"].items():
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids, f"{ref['ref_id']} not in registry for {key}"


def test_each_reference_has_match_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        for ref in entry["references"]:
            metadata = ref["match_metadata"]
            assert metadata["match_type"]
            assert metadata["confidence"]
            assert metadata["notes"]


def test_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.preprocessing.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        leaf = fqdn.removeprefix("sciona.atoms.ml.sklearn.preprocessing.")
        assert leaf in registered, f"{leaf} not in REGISTRY"
