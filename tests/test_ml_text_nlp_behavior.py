from __future__ import annotations

import numpy as np
import pytest


def test_text_nlp_atoms_import() -> None:
    from sciona.atoms.ml import text_nlp

    assert callable(text_nlp.clean_text)
    assert callable(text_nlp.levenshtein)
    assert callable(text_nlp.beam_search)


def test_clean_text_removes_url_and_corrects_spelling() -> None:
    from sciona.atoms.ml.text_nlp import clean_text

    vocab = {"visit": 50, "now": 40}

    assert clean_text("Visitt http://example.com/ Now!", ["url", "lower", "spell"], vocab) == "visit now!"
    assert clean_text("<b>Cafe\u0301</b>", ["html", "unicode", "lower"]) == "café"


def test_string_similarity_atoms() -> None:
    from sciona.atoms.ml.text_nlp import jaro_winkler, levenshtein

    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("sitting", "kitten") == 3
    assert jaro_winkler("martha", "marhta") == pytest.approx(0.9611, abs=1e-4)
    assert jaro_winkler("same", "same") == 1.0


def test_bio_decode_handles_bio_bilou_and_permissive_inside_tags() -> None:
    from sciona.atoms.ml.text_nlp import bio_decode

    tokens = ["John", "lives", "New", "York", "."]
    tags = ["B-PER", "O", "B-LOC", "L-LOC", "O"]

    assert bio_decode(tags, tokens) == [("PER", 0, 0), ("LOC", 2, 3)]
    assert bio_decode(["I-ORG", "I-ORG"], ["Open", "AI"]) == [("ORG", 0, 1)]
    assert bio_decode(["U-PER", "O", "S-LOC"], ["Ada", "in", "NY"]) == [("PER", 0, 0), ("LOC", 2, 2)]


def test_char_to_token_offsets_uses_half_open_character_spans() -> None:
    from sciona.atoms.ml.text_nlp import char_to_token_offsets

    offsets = [(0, 2), (2, 4), (5, 8), (8, 11)]

    assert char_to_token_offsets([(0, 4), (6, 10)], offsets) == [(0, 1), (2, 3)]
    with pytest.raises(Exception):
        char_to_token_offsets([(12, 13)], offsets)


def test_beam_search_returns_sorted_length_bounded_sequences() -> None:
    from sciona.atoms.ml.text_nlp import beam_search

    def log_probs_fn(sequences: tuple[tuple[int, ...], ...]) -> np.ndarray:
        rows = []
        for sequence in sequences:
            if len(sequence) == 1:
                rows.append([-4.0, -5.0, -0.1, -0.2])
            else:
                rows.append([-4.0, -0.1, -2.0, -3.0])
        return np.asarray(rows, dtype=np.float64)

    decoded = beam_search(log_probs_fn, start_token=0, end_token=1, beam_width=2, max_length=4, alpha=0.0)

    assert decoded[0][0] == [0, 2, 1]
    assert len(decoded) == 2
    assert decoded[0][1] >= decoded[1][1]
    assert all(len(sequence) <= 4 for sequence, _ in decoded)


def test_feature_hash_uses_existing_hashing_vectorizer_token() -> None:
    from sciona.atoms.ml.sklearn.feature_extraction import hashing_vectorizer_token
    from sciona.atoms.ml.text_nlp import feature_hash

    tokens = ["apple", "banana", "apple"]
    hashed = feature_hash(tokens, n_features=128)
    apple_column, _ = hashing_vectorizer_token("apple", n_features=128, alternate_sign=False)
    banana_column, _ = hashing_vectorizer_token("banana", n_features=128, alternate_sign=False)

    assert hashed[apple_column] == 2.0
    assert hashed[banana_column] == 1.0
    assert sum(hashed.values()) == 3.0


def test_ngram_and_span_filter_atoms() -> None:
    from sciona.atoms.ml.text_nlp import char_ngrams, filter_spans_by_length, word_ngrams

    assert word_ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    assert char_ngrams("abcd", 3) == ["abc", "bcd"]
    assert filter_spans_by_length([("PER", 0, 2), ("ORG", 4, 5), ("LOC", 6, 10)], {"PER": 3, "ORG": 1}) == [
        ("ORG", 4, 5),
        ("LOC", 6, 10),
    ]


def test_bio_tagging_encoder_decoder() -> None:
    from sciona.atoms.ml.text_nlp import bio_tagging_decoder, bio_tagging_encoder

    # Standard positive path
    spans = [("PER", 0, 1), ("LOC", 3, 3)]
    tags = bio_tagging_encoder(spans, 5)
    assert tags == ["B-PER", "I-PER", "O", "B-LOC", "O"]

    decoded_spans = bio_tagging_decoder(tags)
    assert decoded_spans == [("PER", 0, 1), ("LOC", 3, 3)]

    # Permissive decoding for orphan I-tags
    assert bio_tagging_decoder(["I-PER", "O"]) == [("PER", 0, 0)]
    assert bio_tagging_decoder(["B-PER", "I-LOC"]) == [("PER", 0, 0), ("LOC", 1, 1)]

    # icontract verification
    with pytest.raises(Exception):
        bio_tagging_encoder([("PER", 0, 2), ("LOC", 1, 3)], 5)  # Overlapping

