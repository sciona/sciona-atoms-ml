"""Tokenization atoms backed by Hugging Face fast tokenizer JSON artifacts."""

from __future__ import annotations

from pathlib import Path

import icontract
from tokenizers import Tokenizer

from sciona.ghost.registry import register_atom

from .witnesses import witness_tokenize


def _tokenizer_path_valid(tokenizer_path: Path) -> bool:
    return (
        isinstance(tokenizer_path, Path)
        and tokenizer_path.name.endswith(".json")
        and tokenizer_path.is_file()
    )


def _tokenize_result_valid(result: dict[str, list[int]]) -> bool:
    required_keys = {"input_ids", "attention_mask", "token_type_ids"}
    return (
        isinstance(result, dict)
        and set(result) == required_keys
        and all(isinstance(values, list) for values in result.values())
        and all(isinstance(value, int) and not isinstance(value, bool) for values in result.values() for value in values)
        and len(result["input_ids"]) > 0
        and len(result["attention_mask"]) == len(result["input_ids"])
        and len(result["token_type_ids"]) == len(result["input_ids"])
    )


@register_atom(witness_tokenize)
@icontract.require(
    lambda text: isinstance(text, str) and len(text) > 0,
    "text must be a non-empty string",
)
@icontract.require(
    lambda tokenizer_path: _tokenizer_path_valid(tokenizer_path),
    "tokenizer_path must be an existing tokenizer.json file",
)
@icontract.ensure(
    lambda result: _tokenize_result_valid(result),
    "tokenizer output must contain aligned integer ID, mask, and type-ID lists",
)
def tokenize(text: str, tokenizer_path: Path) -> dict[str, list[int]]:
    """Tokenize text with a local Hugging Face fast tokenizer JSON artifact."""
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoding = tokenizer.encode(text)
    return {
        "input_ids": list(encoding.ids),
        "attention_mask": list(encoding.attention_mask),
        "token_type_ids": list(encoding.type_ids),
    }
