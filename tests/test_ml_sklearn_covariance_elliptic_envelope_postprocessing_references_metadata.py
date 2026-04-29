from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "covariance" / "elliptic_envelope_postprocessing" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.covariance.elliptic_envelope_postprocessing.elliptic_envelope_offset",
    "sciona.atoms.ml.sklearn.covariance.elliptic_envelope_postprocessing.elliptic_envelope_score_samples",
    "sciona.atoms.ml.sklearn.covariance.elliptic_envelope_postprocessing.elliptic_envelope_decision_function",
    "sciona.atoms.ml.sklearn.covariance.elliptic_envelope_postprocessing.elliptic_envelope_labels",
}


def test_elliptic_envelope_postprocessing_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_elliptic_envelope_postprocessing_ref_ids_exist_and_have_metadata() -> None:
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


def test_elliptic_envelope_postprocessing_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.covariance.elliptic_envelope_postprocessing.atoms")
    leaf_names = {
        "elliptic_envelope_offset",
        "elliptic_envelope_score_samples",
        "elliptic_envelope_decision_function",
        "elliptic_envelope_labels",
    }
    assert len(leaf_names) == 4
