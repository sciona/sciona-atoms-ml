"""Ghost witnesses for tokenizer artifact atoms."""

from __future__ import annotations

from pathlib import Path

from sciona.ghost.abstract import AbstractArray


def witness_tokenize(text: str, tokenizer_path: Path) -> dict[str, AbstractArray]:
    """Describe aligned token ID, attention-mask, and type-ID outputs."""
    del tokenizer_path
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    token_count = max(1, len(text.split()))
    ids = AbstractArray(shape=(token_count,), dtype="int64", min_val=0.0)
    return {
        "input_ids": ids,
        "attention_mask": AbstractArray(shape=(token_count,), dtype="int64", min_val=0.0, max_val=1.0),
        "token_type_ids": AbstractArray(shape=(token_count,), dtype="int64", min_val=0.0),
    }
