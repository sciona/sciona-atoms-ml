from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "decomposition" / "sparse_coder_api_shell" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell.sparse_coder_fit_return_self",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell.sparse_coder_transform_dictionary",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell.sparse_coder_requires_fit_tag",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell.sparse_coder_preserves_dtype_tags",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell.sparse_coder_n_features_out",
}


def test_sparse_coder_api_shell_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    fqdn_prefixes = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert fqdn_prefixes == EXPECTED_FQDNS


def test_sparse_coder_api_shell_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for ref_id, meta in payload["atoms"].items():
        assert ref_id.split("@", 1)[0] in EXPECTED_FQDNS
        refs = meta["references"]
        assert refs
        assert refs[0]["ref_id"] == "repo_sklearn"


def test_sparse_coder_api_shell_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell.atoms")
    package = import_module("sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell")
    for leaf_name in {name.rsplit(".", 1)[-1] for name in EXPECTED_FQDNS}:
        assert hasattr(package, leaf_name)
