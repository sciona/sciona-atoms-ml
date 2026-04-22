from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "model_selection" / "diagnostics" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.model_selection.diagnostics.compute_condition_number",
    "sciona.atoms.ml.model_selection.diagnostics.compute_n_p_ratio",
    "sciona.atoms.ml.model_selection.diagnostics.compute_mutual_incoherence",
    "sciona.atoms.ml.model_selection.diagnostics.check_lasso_sample_complexity",
    "sciona.atoms.ml.model_selection.diagnostics.compute_excess_kurtosis",
    "sciona.atoms.ml.model_selection.diagnostics.compute_residual_kurtosis",
    "sciona.atoms.ml.model_selection.diagnostics.compute_dispersion_index",
    "sciona.atoms.ml.model_selection.diagnostics.estimate_tweedie_power",
    "sciona.atoms.ml.model_selection.diagnostics.estimate_noise_level",
    "sciona.atoms.ml.model_selection.diagnostics.count_categorical_features",
    "sciona.atoms.ml.model_selection.diagnostics.compute_skewness",
    "sciona.atoms.ml.model_selection.diagnostics.compute_vif",
    "sciona.atoms.ml.model_selection.diagnostics.test_normality",
    "sciona.atoms.ml.model_selection.diagnostics.is_sparse",
    "sciona.atoms.ml.model_selection.diagnostics.compute_explained_variance_ratio",
    "sciona.atoms.ml.model_selection.diagnostics.check_time_series_index",
}


def test_references_json_exists_and_has_diagnostic_fqdns() -> None:
    assert REFERENCES_PATH.exists()
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_fqdns = {key.partition("@")[0] for key in payload["atoms"]}
    assert EXPECTED_FQDNS.issubset(atom_fqdns)


def test_diagnostic_atoms_have_nonempty_references() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        if key.partition("@")[0] in EXPECTED_FQDNS:
            assert entry["references"], f"empty references for {key}"


def test_diagnostic_ref_ids_exist_in_local_registry() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])
    for key, entry in payload["atoms"].items():
        if key.partition("@")[0] in EXPECTED_FQDNS:
            for ref in entry["references"]:
                assert ref["ref_id"] in registry_ids, f"{ref['ref_id']} not in registry for {key}"


def test_diagnostic_reference_has_match_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        if key.partition("@")[0] in EXPECTED_FQDNS:
            for ref in entry["references"]:
                metadata = ref["match_metadata"]
                assert metadata["match_type"]
                assert metadata["confidence"]
                assert metadata["notes"]


def test_diagnostic_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.model_selection.diagnostics.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        leaf = fqdn.removeprefix("sciona.atoms.ml.model_selection.diagnostics.")
        assert leaf in registered
