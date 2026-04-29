from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "decomposition" / "sparse_coder_transform_shell" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_transform_shell.sparse_coding_transform_alpha",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_transform_shell.sparse_coding_split_sign",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_transform_shell.sparse_coder_n_components",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_transform_shell.sparse_coder_n_features_in",
}


def test_sparse_coder_transform_shell_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_sparse_coder_transform_shell_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for entry in payload["atoms"].values():
        references = entry["references"]
        assert references
        for ref in references:
            assert ref["ref_id"]
            match_metadata = ref["match_metadata"]
            assert match_metadata["match_type"]
            assert match_metadata["confidence"]
            assert match_metadata["notes"]


def test_sparse_coder_transform_shell_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.decomposition.sparse_coder_transform_shell.atoms")
    leaf_names = {
        "sparse_coding_transform_alpha",
        "sparse_coding_split_sign",
        "sparse_coder_n_components",
        "sparse_coder_n_features_in",
    }
    assert len(leaf_names) == 4
