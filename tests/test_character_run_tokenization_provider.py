import json
import pytest
from sciona.atoms.ml.character_run_tokenization import (
    tokenize_character_runs, witness_tokenize_character_runs,
)
from sciona.ghost.registry import REGISTRY


@pytest.mark.parametrize('text,words,spaces,expected', [
    ('ACG?TT', 'ACGT', ' ', ['ACG', '?', 'TT']),
    ('x12+y3', 'xyz0123456789', ' ', ['x12', '+', 'y3']),
    ('αβ α\u0301🙂🙂\r\n', 'αβ', ' \n', ['αβ', ' ', 'α', '\u0301', '🙂', '🙂', '\r', '\n']),
    ('!! \t\na', 'a', ' \t\n', ['!', '!', ' \t\n', 'a']),
    ('', '', '', []),
    ('aab', '', '', ['a', 'a', 'b']),
])
def test_lossless_explicit_domain_policy(text, words, spaces, expected):
    result = tokenize_character_runs(text, words, spaces)
    assert result == expected and ''.join(result) == text
    assert json.loads(json.dumps(result)) == result
    metadata = witness_tokenize_character_runs(text, words, spaces)
    assert metadata['min_tokens'] <= len(result) <= metadata['max_tokens']
    assert metadata['codepoints_preserved'] == len(text)
    assert 'text' not in metadata


def test_membership_and_variable_length_witness():
    assert tokenize_character_runs('aa bb', 'aabb', '  ') == ['aa', ' ', 'bb']
    assert witness_tokenize_character_runs('aaaa', 'a', ' ')['max_tokens'] == 4
    assert 'sciona.atoms.ml.character_run_tokenization.tokenize_character_runs' in REGISTRY


@pytest.mark.parametrize('text,words,spaces', [(None, 'a', ' '), ('a', ['a'], ' '),
                                               ('a', 'a', None), ('a', 'a', 'a')])
def test_invalid_inputs(text, words, spaces):
    for function in (tokenize_character_runs, witness_tokenize_character_runs):
        with pytest.raises(ValueError):
            function(text, words, spaces)
