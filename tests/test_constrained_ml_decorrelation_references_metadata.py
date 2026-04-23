from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "constrained_ml" / "decorrelation" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.constrained_ml.decorrelation.compute_cvm_mass_decorrelation",
    "sciona.atoms.ml.constrained_ml.decorrelation.compute_ks_agreement",
    "sciona.atoms.ml.constrained_ml.decorrelation.roc_auc_truncated_weighted",
    "sciona.atoms.ml.constrained_ml.decorrelation.noise_injection_decorrelation",
}


def test_references_json_exists_and_has_decorrelation_fqdns() -> None:
    assert REFERENCES_PATH.exists()
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_fqdns = {key.partition("@")[0] for key in payload["atoms"]}
    assert EXPECTED_FQDNS.issubset(atom_fqdns)


def test_decorrelation_atoms_have_nonempty_references() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        if key.partition("@")[0] in EXPECTED_FQDNS:
            assert entry["references"], f"empty references for {key}"


def test_decorrelation_ref_ids_exist_in_local_registry() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])
    for key, entry in payload["atoms"].items():
        if key.partition("@")[0] in EXPECTED_FQDNS:
            for ref in entry["references"]:
                assert ref["ref_id"] in registry_ids, f"{ref['ref_id']} not in registry for {key}"


def test_decorrelation_reference_has_match_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        if key.partition("@")[0] in EXPECTED_FQDNS:
            for ref in entry["references"]:
                assert "match_metadata" in ref, f"missing match_metadata in {key}"
                md = ref["match_metadata"]
                assert "confidence" in md
                assert "notes" in md
                assert md["confidence"] in {"high", "medium", "low"}
