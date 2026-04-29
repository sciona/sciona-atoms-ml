from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "decomposition" / "sparse_coder_inverse_shell" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_inverse_shell.sparse_coder_fit_require_matching_features",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_inverse_shell.sparse_coding_expected_code_width",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_inverse_shell.sparse_coding_merge_split_sign",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_inverse_shell.sparse_coder_inverse_transform",
}


def test_sparse_coder_inverse_shell_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_sparse_coder_inverse_shell_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for record in payload["atoms"].values():
        refs = record["references"]
        assert refs
        for ref in refs:
            assert ref["ref_id"]
            assert ref["match_metadata"]["match_type"]
            assert ref["match_metadata"]["confidence"]


def test_sparse_coder_inverse_shell_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.decomposition.sparse_coder_inverse_shell.atoms")
    expected_leaf_names = {fqdn.rsplit(".", 1)[-1] for fqdn in EXPECTED_FQDNS}
    assert len(expected_leaf_names) == len(EXPECTED_FQDNS)
