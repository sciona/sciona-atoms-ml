"""Deterministic text processing and NLP decoding atoms."""

from __future__ import annotations

import bisect
import html
import math
import re
import unicodedata
from collections.abc import Callable

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.atoms.ml.sklearn.feature_extraction import hashing_vectorizer_token
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_beam_search,
    witness_bio_decode,
    witness_char_ngrams,
    witness_char_to_token_offsets,
    witness_clean_text,
    witness_feature_hash,
    witness_filter_spans_by_length,
    witness_jaro_winkler,
    witness_levenshtein,
    witness_qa_span_selector,
    witness_readability_scores,
    witness_word_ngrams,
)

_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WORD_RE = re.compile(r"[A-Za-z]+")
_WHITESPACE_RE = re.compile(r"\s+")
_ALLOWED_CLEAN_OPS = {"html", "url", "unicode", "lower", "spell"}
_VALID_TAG_PREFIXES = {"O", "B", "I", "L", "U", "S", "E"}


def _operations_valid(operations: list[str]) -> bool:
    return bool(
        isinstance(operations, list)
        and len(operations) > 0
        and all(isinstance(operation, str) for operation in operations)
        and set(operations).issubset(_ALLOWED_CLEAN_OPS)
    )


def _vocab_freq_valid(vocab_freq: dict[str, int] | None) -> bool:
    return bool(
        vocab_freq is None
        or (
            isinstance(vocab_freq, dict)
            and len(vocab_freq) > 0
            and all(isinstance(word, str) and word for word in vocab_freq)
            and all(isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in vocab_freq.values())
        )
    )


def _printable_text(text: str) -> bool:
    return all(character.isprintable() or character.isspace() for character in text)


def _known(words: set[str], vocab_freq: dict[str, int]) -> set[str]:
    return {word for word in words if word in vocab_freq}


def _edits1(word: str) -> set[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:index], word[index:]) for index in range(len(word) + 1)]
    deletes = {left + right[1:] for left, right in splits if right}
    transposes = {left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1}
    replaces = {left + letter + right[1:] for left, right in splits if right for letter in letters}
    inserts = {left + letter + right for left, right in splits for letter in letters}
    return deletes | transposes | replaces | inserts


def _spell_candidate(word: str, vocab_freq: dict[str, int]) -> str:
    lowered = word.lower()
    candidates = (
        _known({lowered}, vocab_freq)
        or _known(_edits1(lowered), vocab_freq)
        or _known({edit2 for edit1 in _edits1(lowered) for edit2 in _edits1(edit1)}, vocab_freq)
        or {lowered}
    )
    return max(candidates, key=lambda candidate: (vocab_freq.get(candidate, 0), -len(candidate), candidate))


def _clean_text_result_valid(result: str) -> bool:
    return isinstance(result, str) and _printable_text(result)


@register_atom(witness_clean_text)
@icontract.require(lambda text: isinstance(text, str), "text must be a string")
@icontract.require(lambda operations: _operations_valid(operations), "operations must be selected supported cleaning steps")
@icontract.require(lambda vocab_freq: _vocab_freq_valid(vocab_freq), "vocab_freq must map words to non-negative counts")
@icontract.require(lambda operations, vocab_freq: "spell" not in operations or vocab_freq is not None, "spell correction requires vocab_freq")
@icontract.ensure(lambda result: _clean_text_result_valid(result), "cleaned text must remain printable")
def clean_text(text: str, operations: list[str], vocab_freq: dict[str, int] | None = None) -> str:
    """Normalize one text string with explicit cleaning operations."""
    cleaned = text
    if "html" in operations:
        cleaned = html.unescape(_HTML_RE.sub(" ", cleaned))
    if "url" in operations:
        cleaned = _URL_RE.sub(" ", cleaned)
    if "unicode" in operations:
        cleaned = unicodedata.normalize("NFKC", cleaned)
    if "lower" in operations:
        cleaned = cleaned.lower()
    if "spell" in operations:
        if vocab_freq is None:
            raise ValueError("spell correction requires vocab_freq")
        cleaned = _WORD_RE.sub(lambda match: _spell_candidate(match.group(0), vocab_freq), cleaned)
    cleaned = "".join(character for character in cleaned if character.isprintable() or character.isspace())
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


@register_atom(witness_levenshtein)
@icontract.require(lambda s1: isinstance(s1, str), "s1 must be a string")
@icontract.require(lambda s2: isinstance(s2, str), "s2 must be a string")
@icontract.ensure(lambda result: result >= 0, "edit distance must be non-negative")
@icontract.ensure(lambda result, s1, s2: result <= max(len(s1), len(s2)), "edit distance cannot exceed the longer input")
def levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous = list(range(len(s2) + 1))
    for index1, char1 in enumerate(s1, start=1):
        current = [index1]
        for index2, char2 in enumerate(s2, start=1):
            insert_cost = current[index2 - 1] + 1
            delete_cost = previous[index2] + 1
            replace_cost = previous[index2 - 1] + (char1 != char2)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return int(previous[-1])


def _jaro_similarity(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    match_distance = max(len(s1), len(s2)) // 2 - 1
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    matches = 0
    for index1, char1 in enumerate(s1):
        start = max(0, index1 - match_distance)
        stop = min(index1 + match_distance + 1, len(s2))
        for index2 in range(start, stop):
            if s2_matches[index2] or char1 != s2[index2]:
                continue
            s1_matches[index1] = True
            s2_matches[index2] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    s1_matched = [char for char, matched in zip(s1, s1_matches, strict=True) if matched]
    s2_matched = [char for char, matched in zip(s2, s2_matches, strict=True) if matched]
    transpositions = sum(char1 != char2 for char1, char2 in zip(s1_matched, s2_matched, strict=True)) / 2.0
    return (matches / len(s1) + matches / len(s2) + (matches - transpositions) / matches) / 3.0


@register_atom(witness_jaro_winkler)
@icontract.require(lambda s1: isinstance(s1, str), "s1 must be a string")
@icontract.require(lambda s2: isinstance(s2, str), "s2 must be a string")
@icontract.require(lambda prefix_weight: isinstance(prefix_weight, (int, float)) and not isinstance(prefix_weight, bool) and 0.0 <= float(prefix_weight) <= 0.25, "prefix_weight must be in [0, 0.25]")
@icontract.ensure(lambda result: 0.0 <= result <= 1.0, "similarity must be within [0, 1]")
def jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """Compute the Jaro-Winkler similarity for two strings."""
    jaro = _jaro_similarity(s1, s2)
    prefix = 0
    for char1, char2 in zip(s1[:4], s2[:4], strict=False):
        if char1 != char2:
            break
        prefix += 1
    score = jaro + prefix * float(prefix_weight) * (1.0 - jaro)
    return float(min(1.0, max(0.0, score)))


def _tag_valid(tag: str) -> bool:
    if tag == "O":
        return True
    if "-" not in tag:
        return False
    prefix, label = tag.split("-", 1)
    return bool(prefix in _VALID_TAG_PREFIXES and label)


def _tags_valid(tags: list[str], tokens: list[str]) -> bool:
    return bool(
        isinstance(tags, list)
        and isinstance(tokens, list)
        and len(tags) == len(tokens)
        and all(isinstance(tag, str) and _tag_valid(tag) for tag in tags)
        and all(isinstance(token, str) for token in tokens)
    )


def _spans_ordered(spans: list[tuple[str, int, int]], token_count: int) -> bool:
    previous_start = -1
    for label, start, end in spans:
        if not label or not (0 <= start <= end < token_count):
            return False
        if start <= previous_start:
            return False
        previous_start = start
    return True


@register_atom(witness_bio_decode)
@icontract.require(lambda tags, tokens: _tags_valid(tags, tokens), "tags and tokens must be aligned BIO/BILOU-style strings")
@icontract.ensure(lambda result, tokens: _spans_ordered(result, len(tokens)), "decoded spans must be ordered and inside the token sequence")
def bio_decode(tags: list[str], tokens: list[str]) -> list[tuple[str, int, int]]:
    """Decode BIO, BILOU, or IOBES tags into inclusive token spans."""
    del tokens
    spans: list[tuple[str, int, int]] = []
    active_label: str | None = None
    active_start = 0
    for index, tag in enumerate(tags):
        if tag == "O":
            if active_label is not None:
                spans.append((active_label, active_start, index - 1))
                active_label = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix in {"B", "U", "S"}:
            if active_label is not None:
                spans.append((active_label, active_start, index - 1))
            if prefix in {"U", "S"}:
                spans.append((label, index, index))
                active_label = None
            else:
                active_label = label
                active_start = index
            continue
        if active_label is None or active_label != label:
            if active_label is not None:
                spans.append((active_label, active_start, index - 1))
            active_label = label
            active_start = index
        if prefix in {"L", "E"}:
            spans.append((label, active_start, index))
            active_label = None
    if active_label is not None:
        spans.append((active_label, active_start, len(tags) - 1))
    return spans


def _offset_mapping_valid(offset_mapping: list[tuple[int, int]]) -> bool:
    if not isinstance(offset_mapping, list) or len(offset_mapping) == 0:
        return False
    previous_start = -1
    previous_end = -1
    for start, end in offset_mapping:
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            return False
        if start < previous_start or end < previous_end:
            return False
        previous_start = start
        previous_end = end
    return True


def _char_spans_valid(char_spans: list[tuple[int, int]], offset_mapping: list[tuple[int, int]]) -> bool:
    if not isinstance(char_spans, list) or not _offset_mapping_valid(offset_mapping):
        return False
    starts = [start for start, _ in offset_mapping]
    ends = [end for _, end in offset_mapping]
    for start, end in char_spans:
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or start >= end:
            return False
        first = bisect.bisect_right(ends, start)
        last = bisect.bisect_left(starts, end) - 1
        if first > last or first < 0 or last >= len(offset_mapping):
            return False
    return True


@register_atom(witness_char_to_token_offsets)
@icontract.require(lambda char_spans, offset_mapping: _char_spans_valid(char_spans, offset_mapping), "character spans must overlap monotone token offsets")
@icontract.ensure(lambda result, char_spans: len(result) == len(char_spans), "one token span is returned for each character span")
@icontract.ensure(lambda result, offset_mapping: all(0 <= start <= end < len(offset_mapping) for start, end in result), "token spans must be valid indices")
def char_to_token_offsets(char_spans: list[tuple[int, int]], offset_mapping: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Map half-open character spans to inclusive token-index spans."""
    starts = [start for start, _ in offset_mapping]
    ends = [end for _, end in offset_mapping]
    mapped: list[tuple[int, int]] = []
    for char_start, char_end in char_spans:
        token_start = bisect.bisect_right(ends, char_start)
        token_end = bisect.bisect_left(starts, char_end) - 1
        mapped.append((token_start, token_end))
    return mapped


def _beam_config_valid(start_token: int, end_token: int, beam_width: int, max_length: int, alpha: float) -> bool:
    return bool(
        isinstance(start_token, int)
        and not isinstance(start_token, bool)
        and isinstance(end_token, int)
        and not isinstance(end_token, bool)
        and isinstance(beam_width, int)
        and not isinstance(beam_width, bool)
        and beam_width > 0
        and isinstance(max_length, int)
        and not isinstance(max_length, bool)
        and max_length > 0
        and isinstance(alpha, (int, float))
        and not isinstance(alpha, bool)
        and np.isfinite(float(alpha))
        and float(alpha) >= 0.0
    )


def _length_penalty(length: int, alpha: float) -> float:
    return ((5.0 + float(length)) / 6.0) ** float(alpha)


def _normalized_score(sequence: tuple[int, ...], score: float, alpha: float) -> float:
    return float(score / _length_penalty(max(1, len(sequence)), alpha))


def _beam_result_valid(result: list[tuple[list[int], float]], beam_width: int, max_length: int) -> bool:
    if len(result) > beam_width:
        return False
    previous = math.inf
    for sequence, score in result:
        if not isinstance(sequence, list) or not sequence or len(sequence) > max_length:
            return False
        if not all(isinstance(token, int) and not isinstance(token, bool) for token in sequence):
            return False
        if not isinstance(score, float) or not np.isfinite(score):
            return False
        if score > previous + 1e-12:
            return False
        previous = score
    return True


@register_atom(witness_beam_search)
@icontract.require(lambda start_token, end_token, beam_width, max_length, alpha: _beam_config_valid(start_token, end_token, beam_width, max_length, alpha), "beam-search configuration must be finite and positive where required")
@icontract.ensure(lambda result, beam_width, max_length: _beam_result_valid(result, beam_width, max_length), "beam-search results must be sorted and length-bounded")
def beam_search(
    log_probs_fn: Callable[[tuple[tuple[int, ...], ...]], NDArray[np.float64]],
    start_token: int,
    end_token: int,
    beam_width: int,
    max_length: int,
    alpha: float = 0.7,
) -> list[tuple[list[int], float]]:
    """Decode token sequences with a framework-independent beam search."""
    beams: list[tuple[tuple[int, ...], float, bool]] = [((int(start_token),), 0.0, start_token == end_token)]
    for _ in range(max_length - 1):
        active = [(sequence, score) for sequence, score, done in beams if not done]
        finished = [(sequence, score, done) for sequence, score, done in beams if done]
        if not active:
            break
        active_sequences = tuple(sequence for sequence, _ in active)
        log_probs = np.asarray(log_probs_fn(active_sequences), dtype=np.float64)
        if log_probs.ndim != 2 or log_probs.shape[0] != len(active_sequences) or log_probs.shape[1] == 0 or not np.all(np.isfinite(log_probs)):
            raise ValueError("log_probs_fn must return a finite 2D matrix with one row per active beam")
        candidates: list[tuple[tuple[int, ...], float, bool]] = finished[:]
        for row, (sequence, base_score) in enumerate(active):
            for token in range(log_probs.shape[1]):
                next_sequence = sequence + (int(token),)
                next_score = float(base_score + log_probs[row, token])
                candidates.append((next_sequence, next_score, token == end_token))
        beams = sorted(
            candidates,
            key=lambda item: _normalized_score(item[0], item[1], float(alpha)),
            reverse=True,
        )[:beam_width]
    ranked = sorted(
        ((_normalized_score(sequence, score, float(alpha)), sequence) for sequence, score, _ in beams),
        key=lambda item: item[0],
        reverse=True,
    )[:beam_width]
    return [(list(sequence), float(score)) for score, sequence in ranked]


def _tokens_valid(tokens: list[str]) -> bool:
    return isinstance(tokens, list) and all(isinstance(token, str) and token for token in tokens)


@register_atom(witness_feature_hash)
@icontract.require(lambda tokens: _tokens_valid(tokens), "tokens must be non-empty strings")
@icontract.require(lambda n_features: isinstance(n_features, int) and not isinstance(n_features, bool) and n_features > 0, "n_features must be positive")
@icontract.ensure(lambda result, n_features: all(isinstance(key, int) and 0 <= key < n_features for key in result), "hash keys must be feature indices")
@icontract.ensure(lambda result, tokens: math.isclose(sum(abs(value) for value in result.values()), float(len(tokens))), "hashed counts must preserve token mass")
def feature_hash(tokens: list[str], n_features: int) -> dict[int, float]:
    """Map string tokens to sparse feature-count columns."""
    counts: dict[int, float] = {}
    for token in tokens:
        column, _ = hashing_vectorizer_token(token, n_features=n_features, alternate_sign=False)
        counts[column] = counts.get(column, 0.0) + 1.0
    return counts


@register_atom(witness_word_ngrams)
@icontract.require(lambda tokens: isinstance(tokens, list) and all(isinstance(token, str) for token in tokens), "tokens must be a list of strings")
@icontract.require(lambda n: isinstance(n, int) and not isinstance(n, bool) and n > 0, "n must be positive")
@icontract.ensure(lambda result, tokens, n: len(result) == max(0, len(tokens) - n + 1), "word n-gram count must match the rolling-window count")
@icontract.ensure(lambda result, n: all(len(ngram) == n for ngram in result), "each word n-gram must have length n")
def word_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Generate contiguous token n-grams."""
    return [tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1))]


@register_atom(witness_char_ngrams)
@icontract.require(lambda text: isinstance(text, str), "text must be a string")
@icontract.require(lambda n: isinstance(n, int) and not isinstance(n, bool) and n > 0, "n must be positive")
@icontract.ensure(lambda result, text, n: len(result) == max(0, len(text) - n + 1), "character n-gram count must match the rolling-window count")
@icontract.ensure(lambda result, n: all(len(ngram) == n for ngram in result), "each character n-gram must have length n")
def char_ngrams(text: str, n: int) -> list[str]:
    """Generate contiguous character n-grams."""
    return [text[index : index + n] for index in range(max(0, len(text) - n + 1))]


def _spans_valid(spans: list[tuple[str, int, int]]) -> bool:
    return bool(
        isinstance(spans, list)
        and all(isinstance(label, str) and label and isinstance(start, int) and isinstance(end, int) and start <= end for label, start, end in spans)
    )


def _min_lengths_valid(min_lengths: dict[str, int]) -> bool:
    return bool(
        isinstance(min_lengths, dict)
        and all(isinstance(label, str) and label for label in min_lengths)
        and all(isinstance(length, int) and not isinstance(length, bool) and length > 0 for length in min_lengths.values())
    )


@register_atom(witness_filter_spans_by_length)
@icontract.require(lambda spans: _spans_valid(spans), "spans must be labeled start/end tuples")
@icontract.require(lambda min_lengths: _min_lengths_valid(min_lengths), "minimum lengths must be positive integers")
@icontract.ensure(lambda result, spans: all(span in spans for span in result), "filtered spans must come from the input")
@icontract.ensure(lambda result, min_lengths: all(end - start >= min_lengths.get(label, 0) for label, start, end in result), "filtered spans must meet class length thresholds")
def filter_spans_by_length(spans: list[tuple[str, int, int]], min_lengths: dict[str, int]) -> list[tuple[str, int, int]]:
    """Keep spans whose end-start length meets the class threshold."""
    return [span for span in spans if span[2] - span[1] >= min_lengths.get(span[0], 0)]


def _count_syllables(word: str) -> int:
    """Estimate syllable count for an English word."""
    word = word.lower().strip()
    if len(word) == 0:
        return 0
    # Remove trailing 'e' (silent e)
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]
    # Count vowel groups
    count = len(re.findall(r"[aeiouy]+", word))
    return max(count, 1)


@register_atom(witness_readability_scores)
@icontract.require(lambda text: isinstance(text, str) and len(text.strip()) > 0, "text must be a non-empty string")
@icontract.ensure(
    lambda result: "flesch_kincaid" in result and "smog" in result,
    "result must contain flesch_kincaid and smog keys",
)
@icontract.ensure(
    lambda result: all(math.isfinite(v) for v in result.values()),
    "all scores must be finite",
)
def readability_scores(text: str) -> dict[str, float]:
    """Compute Flesch-Kincaid and SMOG readability indices from text.

    Flesch-Kincaid Reading Ease scores text on a 0-100+ scale where higher
    values indicate easier readability. SMOG estimates the years of education
    needed to understand the text.
    """
    # Tokenize sentences (split on sentence-ending punctuation)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = max(len(sentences), 1)

    # Tokenize words
    words = re.findall(r"[a-zA-Z]+", text)
    n_words = max(len(words), 1)

    # Count syllables and polysyllabic words
    n_syllables = 0
    n_polysyllabic = 0
    for word in words:
        syl = _count_syllables(word)
        n_syllables += syl
        if syl >= 3:
            n_polysyllabic += 1

    # Flesch-Kincaid Reading Ease
    fk = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)

    # SMOG Index
    smog = 3.0 + math.sqrt(n_polysyllabic * 30.0 / n_sentences)

    return {"flesch_kincaid": fk, "smog": smog}


@register_atom(witness_qa_span_selector)
@icontract.require(lambda start_logits, end_logits: len(start_logits) == len(end_logits), "start_logits and end_logits must have equal length")
@icontract.require(lambda max_answer_length: max_answer_length > 0, "max_answer_length must be positive")
@icontract.ensure(lambda result, top_k: len(result) <= top_k, "result must not exceed top_k candidates")
@icontract.ensure(lambda result: all(s <= e for s, e, _ in result), "span start must not exceed span end")
def qa_span_selector(
    start_logits: NDArray[np.float64],
    end_logits: NDArray[np.float64],
    max_answer_length: int = 100,
    top_k: int = 20,
) -> list[tuple[int, int, float]]:
    """Select top-k answer spans from QA start/end logits with length penalty."""
    n = len(start_logits)
    candidates = []
    top_starts = np.argsort(start_logits)[-top_k:][::-1]
    top_ends = np.argsort(end_logits)[-top_k:][::-1]
    for s in top_starts:
        for e in top_ends:
            if e >= s and (e - s + 1) <= max_answer_length:
                score = start_logits[s] + end_logits[e]
                candidates.append((int(s), int(e), float(score)))
    candidates.sort(key=lambda x: -x[2])
    return candidates[:top_k]
