from __future__ import annotations

import json
from pathlib import Path


REFERENCES_PATH = Path("src/sciona/atoms/ml/sklearn/multioutput/chain_fit_order_bookkeeping/references.json")

EXPECTED = {
    "sciona.atoms.ml.sklearn.multioutput.chain_fit_order_bookkeeping.chain_fit_tuple_order_array",
    "sciona.atoms.ml.sklearn.multioutput.chain_fit_order_bookkeeping.chain_fit_require_valid_order",
    "sciona.atoms.ml.sklearn.multioutput.chain_fit_order_bookkeeping.chain_fit_log_message",
}


def test_multioutput_chain_fit_order_bookkeeping_references_cover_atoms() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    found = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert found == EXPECTED


def test_multioutput_chain_fit_order_bookkeeping_references_use_known_sources() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for atom in payload["atoms"].values():
        ref_ids = {reference["ref_id"] for reference in atom["references"]}
        assert "repo_sklearn" in ref_ids
        assert "pedregosa2011sklearn" in ref_ids
