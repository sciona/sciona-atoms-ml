from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "decomposition" / "lda_shell" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.decomposition.lda_shell.lda_doc_topic_prior",
    "sciona.atoms.ml.sklearn.decomposition.lda_shell.lda_topic_word_prior",
    "sciona.atoms.ml.sklearn.decomposition.lda_shell.lda_component_values",
    "sciona.atoms.ml.sklearn.decomposition.lda_shell.lda_normalize_document_topics",
    "sciona.atoms.ml.sklearn.decomposition.lda_shell.lda_perplexity_require_matching_samples",
    "sciona.atoms.ml.sklearn.decomposition.lda_shell.lda_perplexity_require_matching_topics",
    "sciona.atoms.ml.sklearn.decomposition.lda_shell.lda_perplexity_word_count",
    "sciona.atoms.ml.sklearn.decomposition.lda_shell.lda_perplexity_from_bound",
}


def test_lda_shell_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_lda_shell_ref_ids_exist_and_have_metadata() -> None:
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


def test_lda_shell_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.decomposition.lda_shell.atoms")
    leaf_names = {
        "lda_doc_topic_prior",
        "lda_topic_word_prior",
        "lda_component_values",
        "lda_normalize_document_topics",
        "lda_perplexity_require_matching_samples",
        "lda_perplexity_require_matching_topics",
        "lda_perplexity_word_count",
        "lda_perplexity_from_bound",
    }
    assert len(leaf_names) == 8
