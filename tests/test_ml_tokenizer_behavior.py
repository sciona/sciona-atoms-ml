from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


def _write_wordlevel_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


def test_tokenizer_atoms_import() -> None:
    from sciona.atoms.ml import tokenizer

    assert callable(tokenizer.tokenize)


def test_tokenize_uses_local_tokenizer_json_artifact(tmp_path: Path) -> None:
    from sciona.atoms.ml.tokenizer import tokenize

    tokenizer_path = tmp_path / "tokenizer.json"
    _write_wordlevel_tokenizer(tokenizer_path)

    result = tokenize("hello unknown world", tokenizer_path)

    assert result == {
        "input_ids": [1, 0, 2],
        "attention_mask": [1, 1, 1],
        "token_type_ids": [0, 0, 0],
    }
