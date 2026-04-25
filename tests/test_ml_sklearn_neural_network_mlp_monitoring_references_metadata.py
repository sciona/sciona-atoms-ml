from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "neural_network" / "mlp_monitoring" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.neural_network.mlp_monitoring.mlp_monitor_defaults",
    "sciona.atoms.ml.sklearn.neural_network.mlp_monitoring.mlp_monitor_best_loss",
    "sciona.atoms.ml.sklearn.neural_network.mlp_monitoring.mlp_monitor_loss_no_improvement_count",
    "sciona.atoms.ml.sklearn.neural_network.mlp_monitoring.mlp_monitor_best_validation_score",
    "sciona.atoms.ml.sklearn.neural_network.mlp_monitoring.mlp_monitor_validation_no_improvement_count",
}


def test_mlp_monitoring_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_fqdns = {key.partition("@")[0] for key in payload["atoms"]}
    assert atom_fqdns == EXPECTED_FQDNS
    assert "TO_UPDATE" not in REFERENCES_PATH.read_text(encoding="utf-8")


def test_mlp_monitoring_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry_ids = set(json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"])
    for entry in payload["atoms"].values():
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            assert ref["match_metadata"]["match_type"]
            assert ref["match_metadata"]["confidence"]
            assert ref["match_metadata"]["notes"]


def test_mlp_monitoring_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.neural_network.mlp_monitoring.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        assert fqdn.rsplit(".", 1)[-1] in registered
