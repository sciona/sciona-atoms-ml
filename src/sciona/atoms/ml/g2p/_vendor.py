# ---------------------------------------------------------------------------
# Vendored g2p inference from g2p-en 2.1.0
# Original author: Kyubyong Park (kyubyong@gmail.com)
# License: Apache License 2.0
#   https://github.com/Kyubyong/g2p/blob/master/LICENSE
#
# Number normalization adapted from Keith Ito's Tacotron implementation:
#   https://github.com/keithito/tacotron
#   License: MIT
#
# This is a PURE vendoring of the inference path.  It does NOT import
# g2p_en, nltk, or Distance.  The only external dependencies are numpy
# and inflect.
# ---------------------------------------------------------------------------
from __future__ import annotations

import codecs
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

import inflect
import numpy as np
from numpy.typing import NDArray

# ===================================================================== #
#  1.  Number normalisation  (from g2p-en expand.py / keithito tacotron) #
# ===================================================================== #

_inflect_engine = inflect.engine()

_COMMA_NUMBER_RE = re.compile(r"([0-9][0-9,]+[0-9])")
_DECIMAL_NUMBER_RE = re.compile(r"([0-9]+\.[0-9]+)")
_POUNDS_RE = re.compile(r"\xa3([0-9,]*[0-9]+)")
_DOLLARS_RE = re.compile(
    r"\$([0-9.,]*[0-9]+)"
)
_ORDINAL_RE = re.compile(r"[0-9]+(st|nd|rd|th)")
_NUMBER_RE = re.compile(r"[0-9]+")


def _remove_commas(m: re.Match) -> str:
    return m.group(1).replace(",", "")


def _expand_decimal_point(m: re.Match) -> str:
    return m.group(1).replace(".", " point ")


def _expand_dollars(m: re.Match) -> str:
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_pounds(m: re.Match) -> str:
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " pounds"
    pounds = int(parts[0]) if parts[0] else 0
    pence = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if pounds and pence:
        pound_unit = "pound" if pounds == 1 else "pounds"
        pence_unit = "penny" if pence == 1 else "pence"
        return "%s %s, %s %s" % (pounds, pound_unit, pence, pence_unit)
    elif pounds:
        pound_unit = "pound" if pounds == 1 else "pounds"
        return "%s %s" % (pounds, pound_unit)
    elif pence:
        pence_unit = "penny" if pence == 1 else "pence"
        return "%s %s" % (pence, pence_unit)
    else:
        return "zero pounds"


def _expand_ordinal(m: re.Match) -> str:
    return _inflect_engine.number_to_words(m.group(0))


def _expand_number(m: re.Match) -> str:
    num = int(m.group(0))
    if 1000 < num < 3000:
        if num == 2000:
            return "two thousand"
        elif 2000 < num < 2010:
            return "two thousand " + _inflect_engine.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect_engine.number_to_words(num // 100) + " hundred"
        else:
            return _inflect_engine.number_to_words(
                num, andword="", zero="oh", group=2
            )
    else:
        return _inflect_engine.number_to_words(num, andword="")


def normalize_numbers(text: str) -> str:
    """Expand numbers, currencies, and ordinals to spoken-form words."""
    text = _COMMA_NUMBER_RE.sub(_remove_commas, text)
    text = _POUNDS_RE.sub(_expand_pounds, text)
    text = _DOLLARS_RE.sub(_expand_dollars, text)
    text = _DECIMAL_NUMBER_RE.sub(_expand_decimal_point, text)
    text = _ORDINAL_RE.sub(_expand_ordinal, text)
    text = _NUMBER_RE.sub(_expand_number, text)
    return text


# ===================================================================== #
#  2.  Homograph dictionary loader                                       #
# ===================================================================== #

def load_homographs(
    path: Path,
) -> dict[str, tuple[list[str], list[str], str]]:
    """Load homographs.en into ``{word: (pron1, pron2, pos1)}``.

    File format (lines starting with ``#`` are comments)::

        HEADWORD|PRONUNCIATION1|PRONUNCIATION2|POS
    """
    result: dict[str, tuple[list[str], list[str], str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            headword = parts[0].strip().lower()
            pron1 = parts[1].strip().split()
            pron2 = parts[2].strip().split()
            pos = parts[3].strip()
            result[headword] = (pron1, pron2, pos)
    return result


# ===================================================================== #
#  3.  CMUDict loader                                                    #
# ===================================================================== #

def load_cmudict(path: Path) -> dict[str, list[list[str]]]:
    """Load CMU pronunciation dictionary.

    *path* may point to:

    * A plain-text ``cmudict`` file (lines like ``WORD  PH1 PH2 PH3``).
    * An NLTK corpus zip (``cmudict.zip``) containing the text file inside
      ``cmudict/cmudict``.
    * An NLTK corpus directory (``cmudict/``) containing the text file.

    Lines beginning with ``;;;`` are comments.  Alternate pronunciations
    are indicated by ``WORD(1)``, ``WORD(2)``, etc.
    """
    # Resolve to an iterable of text lines.
    lines: list[str]
    p = Path(path)

    if p.is_dir():
        # NLTK-style corpus directory — look for the plain-text file inside.
        candidates = [p / "cmudict", p / "cmudict.dict"]
        txt_path = next((c for c in candidates if c.exists()), None)
        if txt_path is None:
            raise FileNotFoundError(
                f"Could not find cmudict text file inside directory {p}"
            )
        lines = txt_path.read_text(encoding="latin-1").splitlines()
    elif p.suffix == ".zip":
        with zipfile.ZipFile(p) as zf:
            # Try common inner paths.
            for inner in ("cmudict/cmudict", "cmudict/cmudict.dict", "cmudict"):
                if inner in zf.namelist():
                    with zf.open(inner) as f:
                        raw = f.read().decode("latin-1")
                    lines = raw.splitlines()
                    break
            else:
                raise FileNotFoundError(
                    f"Could not find cmudict text file inside zip {p}"
                )
    else:
        lines = p.read_text(encoding="latin-1").splitlines()

    result: dict[str, list[list[str]]] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith(";;;"):
            continue
        # Data format: WORD  PH1 PH2 PH3  (two-space separator)
        if "  " not in line:
            continue
        word_part, phones_part = line.split("  ", 1)
        # Strip variant number: WORD(1) -> WORD
        base_word = re.sub(r"\(\d+\)$", "", word_part).strip().lower()
        phonemes = phones_part.strip().split()
        result.setdefault(base_word, []).append(phonemes)
    return result


# ===================================================================== #
#  4.  Averaged perceptron POS tagger                                    #
# ===================================================================== #

class PerceptronTagger:
    """Minimal averaged-perceptron POS tagger compatible with NLTK's
    ``averaged_perceptron_tagger_eng`` JSON asset files.
    """

    START = ["-START-", "-START2-"]

    def __init__(
        self,
        weights: dict[str, dict[str, float]],
        classes: list[str],
        tagdict: dict[str, str],
    ) -> None:
        self.weights = weights
        self.classes = set(classes)
        self.tagdict = tagdict

    # ------------------------------------------------------------------
    #  Feature extraction (mirrors NLTK PerceptronTagger._get_features)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_features(
        i: int,
        word: str,
        context: list[str],
        prev: str,
        prev2: str,
    ) -> dict[str, int]:
        features: dict[str, int] = {}

        def add(name: str, *args: str) -> None:
            features[" ".join((name,) + args)] = 1

        i_suffix = word[-3:]
        add("bias")
        add("i suffix", i_suffix)
        add("i pref1", word[0] if word else "")
        add("i-1 tag", prev)
        add("i-2 tag", prev2)
        add("i tag+i-2 tag", prev, prev2)
        add("i word", context[i])
        add("i-1 tag+i word", prev, context[i])
        add("i-1 word", context[i - 1] if i > 0 else "")
        add("i+1 word", context[i + 1] if i + 1 < len(context) else "")
        add("i-1 suffix", context[i - 1][-3:] if i > 0 else "")
        add("i-2 word", context[i - 2] if i - 2 >= 0 else "")
        add("i+2 word", context[i + 2] if i + 2 < len(context) else "")
        return features

    # ------------------------------------------------------------------
    #  Predict best tag for a single token
    # ------------------------------------------------------------------

    def _predict(self, features: dict[str, int]) -> str:
        scores: dict[str, float] = {}
        for feat in features:
            if feat not in self.weights:
                continue
            tag_weights = self.weights[feat]
            for tag, weight in tag_weights.items():
                scores[tag] = scores.get(tag, 0.0) + weight
        if not scores:
            return "NN"  # default fallback
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def tag(self, words: list[str]) -> list[tuple[str, str]]:
        """Tag *words* and return ``[(word, tag), ...]``."""
        return self.pos_tag(words)

    def pos_tag(self, words: list[str]) -> list[tuple[str, str]]:
        """Tag *words* and return ``[(word, tag), ...]``."""
        prev, prev2 = self.START
        context = self.START + [self._normalize(w) for w in words] + ["-END-"]
        output: list[tuple[str, str]] = []
        for i, word in enumerate(words):
            # Fast path: known unambiguous word
            tag = self.tagdict.get(word)
            if tag is None:
                features = self._get_features(
                    i + len(self.START),
                    self._normalize(word),
                    context,
                    prev,
                    prev2,
                )
                tag = self._predict(features)
            output.append((word, tag))
            prev2 = prev
            prev = tag
        return output

    @staticmethod
    def _normalize(word: str) -> str:
        if "-" in word and word[0] != "-":
            return "!HYPHEN"
        if word.isdigit() and len(word) == 4:
            return "!YEAR"
        if word and word[0].isdigit():
            return "!DIGITS"
        return word.lower()


def load_perceptron_tagger(tagger_dir: Path) -> PerceptronTagger:
    """Load NLTK averaged perceptron tagger from JSON asset files.

    Expected files inside *tagger_dir*:

    * ``averaged_perceptron_tagger_eng.classes.json``
    * ``averaged_perceptron_tagger_eng.tagdict.json``
    * ``averaged_perceptron_tagger_eng.weights.json``
    """
    tagger_dir = Path(tagger_dir)

    def _load_json(name: str) -> Any:
        fp = tagger_dir / name
        with open(fp, encoding="utf-8") as fh:
            return json.load(fh)

    classes: list[str] = _load_json(
        "averaged_perceptron_tagger_eng.classes.json"
    )
    tagdict: dict[str, str] = _load_json(
        "averaged_perceptron_tagger_eng.tagdict.json"
    )
    weights: dict[str, dict[str, float]] = _load_json(
        "averaged_perceptron_tagger_eng.weights.json"
    )
    return PerceptronTagger(weights=weights, classes=classes, tagdict=tagdict)


# ===================================================================== #
#  5.  GRU encoder-decoder for OOV prediction                           #
# ===================================================================== #

GRAPHEMES = list("<pad> <unk> </s> a b c d e f g h i j k l m n o p q r s t u v w x y z".split())
PHONEMES = [
    "<pad>", "<unk>", "<s>", "</s>",
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2",
    "AH0", "AH1", "AH2", "AO0", "AO1", "AO2",
    "AW0", "AW1", "AW2", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH",
    "EH0", "EH1", "EH2", "ER0", "ER1", "ER2",
    "EY0", "EY1", "EY2",
    "F", "G", "HH",
    "IH0", "IH1", "IH2", "IY0", "IY1", "IY2",
    "JH", "K", "L", "M", "N", "NG",
    "OW0", "OW1", "OW2", "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH",
    "UH0", "UH1", "UH2", "UW0", "UW1", "UW2",
    "V", "W", "Y", "Z", "ZH",
]

_G2IDX = {g: i for i, g in enumerate(GRAPHEMES)}
_IDX2P = {i: p for i, p in enumerate(PHONEMES)}


def load_checkpoint(path: Path) -> dict[str, NDArray]:
    """Load ``checkpoint20.npz`` with ``allow_pickle=False``."""
    return dict(np.load(str(path), allow_pickle=False))


def _gru_cell(
    x: NDArray,
    h: NDArray,
    w_ih: NDArray,
    w_hh: NDArray,
    b_ih: NDArray,
    b_hh: NDArray,
) -> NDArray:
    """Single GRU cell forward pass matching g2p-en's convention.

    Weight layout: ``w_ih`` is ``(3*hidden, input_dim)`` — transposed in matmul.
    The three gates (r, z, n) are concatenated along axis 0.
    """
    rzn_ih = np.matmul(x, w_ih.T) + b_ih
    rzn_hh = np.matmul(h, w_hh.T) + b_hh

    two_thirds = rzn_ih.shape[-1] * 2 // 3
    rz_ih, n_ih = rzn_ih[:, :two_thirds], rzn_ih[:, two_thirds:]
    rz_hh, n_hh = rzn_hh[:, :two_thirds], rzn_hh[:, two_thirds:]

    rz = _sigmoid(rz_ih + rz_hh)
    r, z = np.split(rz, 2, -1)
    n = np.tanh(n_ih + r * n_hh)
    return (1 - z) * n + z * h


def _sigmoid(x: NDArray) -> NDArray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_oov(word: str, variables: dict[str, NDArray]) -> list[str]:
    """Predict phonemes for an out-of-vocabulary word using the GRU
    encoder-decoder from g2p-en.

    *variables* is the dict returned by :func:`load_checkpoint`.
    The checkpoint uses PyTorch-style variable names:
    ``enc_emb``, ``enc_w_ih``, ``enc_w_hh``, ``enc_b_ih``, ``enc_b_hh``,
    ``dec_emb``, ``dec_w_ih``, ``dec_w_hh``, ``dec_b_ih``, ``dec_b_hh``,
    ``fc_w``, ``fc_b``.

    Weight shapes: ``w_ih`` is ``(3*hidden, input_dim)``,
    ``w_hh`` is ``(3*hidden, hidden)``.  The matmul in the original code
    transposes: ``np.matmul(x, w_ih.T)``.
    """
    unk_idx = _G2IDX["<unk>"]
    eos_idx = _G2IDX["</s>"]
    x_ids = [_G2IDX.get(ch, unk_idx) for ch in word.lower()] + [eos_idx]

    enc_emb = variables["enc_emb"]
    enc_w_ih = variables["enc_w_ih"]
    enc_w_hh = variables["enc_w_hh"]
    enc_b_ih = variables["enc_b_ih"]
    enc_b_hh = variables["enc_b_hh"]

    dec_emb = variables["dec_emb"]
    dec_w_ih = variables["dec_w_ih"]
    dec_w_hh = variables["dec_w_hh"]
    dec_b_ih = variables["dec_b_ih"]
    dec_b_hh = variables["dec_b_hh"]
    fc_w = variables["fc_w"]
    fc_b = variables["fc_b"]

    hidden_dim = enc_w_hh.shape[1]

    # --- Encoder ---
    x = np.take(enc_emb, np.array(x_ids), axis=0)
    x = np.expand_dims(x, 0)  # (1, T, emb)
    h = np.zeros((1, hidden_dim), dtype=np.float32)
    for t in range(len(x_ids)):
        h = _gru_cell(x[:, t, :], h, enc_w_ih, enc_w_hh, enc_b_ih, enc_b_hh)
    last_hidden = h

    # --- Decoder ---
    dec = np.take(dec_emb, [2], axis=0)  # 2 = <s>
    h = last_hidden
    preds: list[int] = []
    for _ in range(20):
        h = _gru_cell(dec, h, dec_w_ih, dec_w_hh, dec_b_ih, dec_b_hh)
        logits = np.matmul(h, fc_w.T) + fc_b
        pred = int(logits.argmax())
        if pred == 3:  # 3 = </s>
            break
        preds.append(pred)
        dec = np.take(dec_emb, [pred], axis=0)

    return [_IDX2P.get(idx, "<unk>") for idx in preds]


# ===================================================================== #
#  6.  Tokenizer (replaces NLTK TweetTokenizer)                         #
# ===================================================================== #

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,?!\-]")


def tokenize(text: str) -> list[str]:
    """Simple regex tokenizer for cleaned, lowercased g2p input."""
    return _TOKEN_RE.findall(text)
