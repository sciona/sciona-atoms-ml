"""Ghost witness for the g2p_convert atom."""

from __future__ import annotations

from pathlib import Path


def witness_g2p_convert(
    text: str,
    cmudict_path: Path,
    tagger_dir: Path,
    checkpoint_path: Path,
    homographs_path: Path,
) -> list[str]:
    """Describe a phoneme sequence output."""
    del cmudict_path, tagger_dir, checkpoint_path, homographs_path
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    return ["AH0"]
