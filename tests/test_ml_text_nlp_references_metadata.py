from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "text_nlp" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.text_nlp.clean_text",
    "sciona.atoms.ml.text_nlp.levenshtein",
    "sciona.atoms.ml.text_nlp.jaro_winkler",
    "sciona.atoms.ml.text_nlp.bio_decode",
    "sciona.atoms.ml.text_nlp.char_to_token_offsets",
    "sciona.atoms.ml.text_nlp.beam_search",
    "sciona.atoms.ml.text_nlp.feature_hash",
    "sciona.atoms.ml.text_nlp.word_ngrams",
    "sciona.atoms.ml.text_nlp.char_ngrams",
    "sciona.atoms.ml.text_nlp.filter_spans_by_length",
}


def test_text_nlp_references_json_exists_and_has_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))

    assert EXPECTED_FQDNS == {key.partition("@")[0] for key in payload["atoms"]}


def test_text_nlp_ref_ids_exist_in_local_registry() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])

    for entry in payload["atoms"].values():
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            assert ref["match_metadata"]["match_type"]
            assert ref["match_metadata"]["confidence"]
            assert ref["match_metadata"]["notes"]


def test_text_nlp_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.text_nlp.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}

    for fqdn in EXPECTED_FQDNS:
        assert fqdn.removeprefix("sciona.atoms.ml.text_nlp.") in registered
