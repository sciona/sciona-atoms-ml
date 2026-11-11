"""Lossless code-point tokenization with caller-defined character policy."""
from sciona.ghost.registry import register_atom


def _policy(word_characters, whitespace_characters):
    from sciona._runtime.ml.character_run_tokenizer import CharacterRunTokenizer
    return CharacterRunTokenizer(word_characters, whitespace_characters)


def witness_tokenize_character_runs(text: str, word_characters: str,
                                    whitespace_characters: str) -> dict:
    """Describe bounds only; token count depends on the actual character runs.

    This metadata is not an exact-shape token array and contains no text.
    Downstream witnesses must accept variable-length token sequences.
    """
    if not isinstance(text, str):
        raise ValueError('Text input required')
    _policy(word_characters, whitespace_characters)
    return dict(format='sciona.character_run_tokens.abstract.v1',
                element_type='str', min_tokens=int(bool(text)), max_tokens=len(text),
                codepoints_preserved=len(text), lossless=True)


@register_atom(witness_tokenize_character_runs)
def tokenize_character_runs(text: str, word_characters: str,
                            whitespace_characters: str) -> list:
    """Split text into word runs, whitespace runs and individual other code points.

    Alphabets are explicit disjoint strings; repeated alphabet characters do
    not affect membership. Empty alphabets and empty input are supported.
    Joining output tokens exactly recovers input. Punctuation is never merged
    unless included in a run alphabet. No case folding, Unicode normalization,
    stemming, transliteration or whitespace deletion occurs. Combining marks
    and emoji are code points, not grapheme clusters. Runs are local to a whole
    input string; independent chunk tokenization need not preserve boundaries.
    Caller selects domain policy and maintains it across training/inference.
    Executes serially with no fitted state or worker processes.
    """
    return _policy(word_characters, whitespace_characters)(text)
