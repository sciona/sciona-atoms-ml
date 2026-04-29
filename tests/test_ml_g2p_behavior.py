from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Asset paths (same as test_g2p.py)
# ---------------------------------------------------------------------------
CMUDICT_ZIP = Path("/tmp/sciona-g2p-minimal-zipeng-data/corpora/cmudict.zip")
TAGGER_DIR = Path("/tmp/sciona-g2p-nltk-run2-data/taggers/averaged_perceptron_tagger_eng")
CHECKPOINT = Path("/tmp/sciona-g2p-wheel-unpacked/g2p_en/checkpoint20.npz")
HOMOGRAPHS = Path("/tmp/sciona-g2p-wheel-unpacked/g2p_en/homographs.txt")

ASSETS_AVAILABLE = all(p.exists() for p in [CMUDICT_ZIP, TAGGER_DIR, CHECKPOINT, HOMOGRAPHS])

skip_no_assets = pytest.mark.skipif(
    not ASSETS_AVAILABLE,
    reason="G2P assets not available (see G2P.md for download instructions)",
)


def test_g2p_convert_imports_successfully() -> None:
    from sciona.atoms.ml.g2p import g2p_convert

    assert callable(g2p_convert)


def test_vendor_module_imports_without_nltk_or_g2p_en() -> None:
    from sciona.atoms.ml.g2p import _vendor

    assert hasattr(_vendor, "normalize_numbers")
    assert hasattr(_vendor, "load_homographs")
    assert hasattr(_vendor, "tokenize")
    assert hasattr(_vendor, "PerceptronTagger")


def test_normalize_numbers_on_simple_strings() -> None:
    from sciona.atoms.ml.g2p._vendor import normalize_numbers

    assert "two hundred" in normalize_numbers("200")
    assert "dollar" in normalize_numbers("$1")
    assert "point" in normalize_numbers("3.14")


def test_load_homographs_on_inline_fixture() -> None:
    from sciona.atoms.ml.g2p._vendor import load_homographs

    content = (
        "# comment line\n"
        "refuse|R IH0 F Y UW1 Z|R EH1 F Y UW2 Z|VB\n"
        "lead|L IY1 D|L EH1 D|VB\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
        fh.write(content)
        fh.flush()
        result = load_homographs(Path(fh.name))

    assert "refuse" in result
    assert "lead" in result
    assert result["refuse"][0] == ["R", "IH0", "F", "Y", "UW1", "Z"]
    assert result["refuse"][2] == "VB"


def test_tokenize_on_cleaned_text() -> None:
    from sciona.atoms.ml.g2p._vendor import tokenize

    tokens = tokenize("hello world, i'm here.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "i'm" in tokens
    assert "." in tokens


def test_perceptron_tagger_with_empty_weights() -> None:
    from sciona.atoms.ml.g2p._vendor import PerceptronTagger

    tagger = PerceptronTagger(weights={}, classes=[], tagdict={})
    result = tagger.tag([])
    assert result == []
