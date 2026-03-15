"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.extensions.text.unigram import UnigramTokenizer, Trie
import json


def test_trie_basic():
    """Provides semantic functionality and verification."""
    trie = Trie()
    trie.insert("a", -1.0, 1)
    trie.insert("ab", -2.0, 2)
    trie.insert("c", -3.0, 3)
    prefs = trie.get_prefixes("abc", 0)
    assert len(prefs) == 2
    assert prefs[0] == (1, -1.0, 1)
    assert prefs[1] == (2, -2.0, 2)


def test_unigram_tokenizer_basic():
    """Provides semantic functionality and verification."""
    vocab = {"[UNK]": 0, "a": 1, "b": 2, "ab": 3}
    scores = {"[UNK]": 0.0, "a": -1.0, "b": -1.0, "ab": -1.5}
    tokenizer = UnigramTokenizer(vocab, scores, unk_score=-100.0)
    ids = tokenizer.encode("ab")
    assert ids == [3]
    ids2 = tokenizer.encode("aba")
    assert ids2 == [3, 1]
    assert tokenizer.decode([3, 1]) == "aba"


def test_unigram_tokenizer_unk():
    """Provides semantic functionality and verification."""
    vocab = {"[UNK]": 0, "a": 1}
    scores = {"[UNK]": 0.0, "a": -1.0}
    tokenizer = UnigramTokenizer(vocab, scores, unk_score=-100.0)
    ids = tokenizer.encode("ab")
    assert ids == [1, 0]


def test_unigram_empty_encode():
    """Provides semantic functionality and verification."""
    vocab = {"[UNK]": 0}
    scores = {"[UNK]": 0.0}
    tokenizer = UnigramTokenizer(vocab, scores)
    assert tokenizer.encode("") == []


def test_unigram_from_huggingface(tmpdir):
    """Provides semantic functionality and verification."""
    data = {"model": {"vocab": [["[UNK]", 0.0], [" a", -1.0], [" b", -2.0]]}}
    file = tmpdir.join("tokenizer.json")
    with open(file, "w") as f:
        json.dump(data, f)
    tokenizer = UnigramTokenizer.from_huggingface(str(file))
    assert tokenizer.vocab[" a"] == 1
    assert tokenizer.scores[" a"] == -1.0
    tokenizer2 = UnigramTokenizer.from_huggingface(data)
    assert tokenizer2.vocab[" b"] == 2
