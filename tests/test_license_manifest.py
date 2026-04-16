from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "licenses" / "provider_license.json"


def test_ml_license_manifest_has_verified_sklearn_override() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())

    assert manifest["provider_repo"] == "sciona-atoms-ml"
    assert manifest["repo_default_license"]["license_spdx"] == "NOASSERTION"
    assert manifest["repo_default_license"]["status"] == "unknown"

    overrides = manifest["family_overrides"]
    assert len(overrides) == 1
    override = overrides[0]
    assert override["family"] == "sciona.atoms.ml.sklearn.images"
    assert override["license_spdx"] == "BSD-3-Clause"
    assert override["status"] == "verified"
    assert override["review_record_path"] == "docs/license-records/sklearn_images.md"
    assert override["authoritative_sources"][0]["source_url"] == "https://github.com/scikit-learn/scikit-learn"

    unresolved = {row["family"] for row in manifest["unresolved_families"]}
    assert unresolved == {"sciona.atoms.ml.datadriven"}

    assert (REPO_ROOT / override["review_record_path"]).exists()
