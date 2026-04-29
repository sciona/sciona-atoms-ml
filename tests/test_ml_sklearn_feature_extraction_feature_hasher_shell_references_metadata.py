from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "feature_extraction" / "feature_hasher_shell" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.feature_extraction.feature_hasher_shell.feature_hasher_dict_items",
    "sciona.atoms.ml.sklearn.feature_extraction.feature_hasher_shell.feature_hasher_pair_items",
    "sciona.atoms.ml.sklearn.feature_extraction.feature_hasher_shell.feature_hasher_string_items",
    "sciona.atoms.ml.sklearn.feature_extraction.feature_hasher_shell.feature_hasher_sample_count",
    "sciona.atoms.ml.sklearn.feature_extraction.feature_hasher_shell.feature_hasher_require_nonempty_samples",
    "sciona.atoms.ml.sklearn.feature_extraction.feature_hasher_shell.feature_hasher_csr_matrix",
}


def test_feature_hasher_shell_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_feature_hasher_shell_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for record in payload["atoms"].values():
        refs = record["references"]
        assert refs
        for ref in refs:
            assert ref["ref_id"]
            assert ref["match_metadata"]["match_type"]
            assert ref["match_metadata"]["confidence"]


def test_feature_hasher_shell_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.feature_extraction.feature_hasher_shell.atoms")
    expected_leaf_names = {fqdn.rsplit(".", 1)[-1] for fqdn in EXPECTED_FQDNS}
    assert len(expected_leaf_names) == len(EXPECTED_FQDNS)
