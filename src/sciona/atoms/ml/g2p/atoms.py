"""Grapheme-to-phoneme conversion atom with explicit stateful resource injection."""

from __future__ import annotations

from pathlib import Path

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import witness_g2p_convert


def _assets_valid(
    cmudict_path: Path,
    tagger_dir: Path,
    checkpoint_path: Path,
    homographs_path: Path,
) -> bool:
    return (
        isinstance(cmudict_path, Path)
        and isinstance(tagger_dir, Path)
        and isinstance(checkpoint_path, Path)
        and isinstance(homographs_path, Path)
    )


@register_atom(witness_g2p_convert)
@icontract.require(
    lambda text: isinstance(text, str) and len(text) > 0,
    "text must be a non-empty string",
)
@icontract.require(
    lambda cmudict_path, tagger_dir, checkpoint_path, homographs_path: _assets_valid(
        cmudict_path, tagger_dir, checkpoint_path, homographs_path
    ),
    "all asset paths must be Path objects",
)
@icontract.ensure(
    lambda result: isinstance(result, list) and all(isinstance(p, str) for p in result),
    "result must be a list of phoneme strings",
)
def g2p_convert(
    text: str,
    cmudict_path: Path,
    tagger_dir: Path,
    checkpoint_path: Path,
    homographs_path: Path,
) -> list[str]:
    """Convert English text to ARPAbet phonemes.

    Uses CMUDict for known words, a perceptron POS tagger for homograph
    disambiguation, and a GRU seq2seq model for out-of-vocabulary prediction.
    All resources are loaded from explicit file paths — no network access.
    """
    from ._vendor import (
        load_checkpoint,
        load_cmudict,
        load_homographs,
        load_perceptron_tagger,
        normalize_numbers,
        predict_oov,
        tokenize,
        GRAPHEMES,
        PHONEMES,
    )
    import re
    import unicodedata

    # Load resources
    cmu = load_cmudict(cmudict_path)
    tagger = load_perceptron_tagger(tagger_dir)
    variables = load_checkpoint(checkpoint_path)
    homographs = load_homographs(homographs_path)

    # Preprocessing (from g2p-en)
    processed = normalize_numbers(text)
    processed = "".join(
        char
        for char in unicodedata.normalize("NFD", processed)
        if unicodedata.category(char) != "Mn"
    )
    processed = processed.lower()
    processed = re.sub("[^ a-z'.,?!\\-]", "", processed)
    processed = processed.replace("i.e.", "that is")
    processed = processed.replace("e.g.", "for example")

    # Tokenize and POS tag
    words = tokenize(processed)
    tokens = tagger.tag(words)

    # G2P conversion
    prons: list[str] = []
    for word, pos in tokens:
        if re.search("[a-z]", word) is None:
            pron = [word]
        elif word in homographs:
            pron1, pron2, pos1 = homographs[word]
            if pos.startswith(pos1):
                pron = pron1
            else:
                pron = pron2
        elif word in cmu:
            pron = cmu[word][0]
        else:
            pron = predict_oov(word, variables)

        prons.extend(pron)
        prons.extend([" "])

    return prons[:-1] if prons else prons
