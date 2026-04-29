"""Tests for the g2p_convert atom with real vendored assets."""

from __future__ import annotations

from pathlib import Path

import pytest

from sciona.atoms.ml.g2p import g2p_convert

# ---------------------------------------------------------------------------
# Asset paths
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


@skip_no_assets
def test_g2p_convert_known_word():
    """Known CMU word produces dictionary pronunciation."""
    result = g2p_convert(
        "hello",
        cmudict_path=CMUDICT_ZIP,
        tagger_dir=TAGGER_DIR,
        checkpoint_path=CHECKPOINT,
        homographs_path=HOMOGRAPHS,
    )
    phoneme_str = " ".join(result)
    # "hello" -> HH AH0 L OW1 or HH EH0 L OW1 in CMUDict
    assert "HH" in phoneme_str


@skip_no_assets
def test_g2p_convert_homograph_disambiguation():
    """Homograph 'refuse' gets different pronunciation by POS."""
    result = g2p_convert(
        "I refuse to collect the refuse around here.",
        cmudict_path=CMUDICT_ZIP,
        tagger_dir=TAGGER_DIR,
        checkpoint_path=CHECKPOINT,
        homographs_path=HOMOGRAPHS,
    )
    phoneme_str = " ".join(result)
    # First "refuse" (verb) -> R IH0 F Y UW1 Z
    # Second "refuse" (noun) -> R EH1 F Y UW2 Z
    assert "IH0" in phoneme_str, f"Expected verb form IH0 in: {phoneme_str}"
    assert "EH1" in phoneme_str, f"Expected noun form EH1 in: {phoneme_str}"


@skip_no_assets
def test_g2p_convert_oov_word():
    """Out-of-vocabulary word produces non-empty prediction via neural model."""
    result = g2p_convert(
        "activationist",
        cmudict_path=CMUDICT_ZIP,
        tagger_dir=TAGGER_DIR,
        checkpoint_path=CHECKPOINT,
        homographs_path=HOMOGRAPHS,
    )
    # Filter out whitespace-only tokens
    phonemes = [p for p in result if p.strip()]
    assert len(phonemes) > 0, "OOV word should produce at least one phoneme"


@skip_no_assets
def test_g2p_convert_deterministic():
    """Same input produces identical output across runs."""
    kwargs = dict(
        cmudict_path=CMUDICT_ZIP,
        tagger_dir=TAGGER_DIR,
        checkpoint_path=CHECKPOINT,
        homographs_path=HOMOGRAPHS,
    )
    result1 = g2p_convert("hello world", **kwargs)
    result2 = g2p_convert("hello world", **kwargs)
    assert result1 == result2


@skip_no_assets
def test_g2p_convert_number_expansion():
    """Numbers are expanded to words before conversion."""
    result = g2p_convert(
        "I have $250 in my pocket.",
        cmudict_path=CMUDICT_ZIP,
        tagger_dir=TAGGER_DIR,
        checkpoint_path=CHECKPOINT,
        homographs_path=HOMOGRAPHS,
    )
    # "$250" should expand to "two hundred fifty dollars" -- non-trivial output
    assert len(result) > 5, f"Expected substantial output for number expansion, got {result}"
