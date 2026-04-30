from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "decomposition" / "lda_bound" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.decomposition.lda_bound.lda_dirichlet_loglikelihood",
    "sciona.atoms.ml.sklearn.decomposition.lda_bound.lda_document_log_probability_bound",
    "sciona.atoms.ml.sklearn.decomposition.lda_bound.lda_apply_subsampling_ratio",
    "sciona.atoms.ml.sklearn.decomposition.lda_bound.lda_approx_bound_from_expectations",
}


def test_lda_bound_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_lda_bound_ref_ids_exist_and_have_metadata() -> None:
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


def test_lda_bound_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.decomposition.lda_bound.atoms")
    leaf_names = {
        "lda_dirichlet_loglikelihood",
        "lda_document_log_probability_bound",
        "lda_apply_subsampling_ratio",
        "lda_approx_bound_from_expectations",
    }
    assert len(leaf_names) == 4

