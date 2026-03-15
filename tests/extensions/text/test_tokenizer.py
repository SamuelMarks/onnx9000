"""Module providing core logic and structural definitions."""

import json
import pytest
from onnx9000.extensions.text.tokenizer import (
    BPETokenizer,
    Tokenizer,
    WordPieceTokenizer,
)


def test_bpe_tokenizer_basic():
    """Provides semantic functionality and verification."""
    vocab = {"[UNK]": 0, "hel": 1, "lo": 2, " world": 3, "hello": 4}
    merges = {("h", "e"): 1, ("he", "l"): 2, ("hel", "l"): 3, ("hell", "o"): 4}
    tokenizer = BPETokenizer(vocab, merges)
    ids = tokenizer.encode("hello")
    assert ids == [4]
    assert tokenizer.decode([4, 3]) == "hello world"


def test_bpe_tokenizer_edge_cases():
    """Provides semantic functionality and verification."""
    vocab = {
        "[UNK]": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "ab": 4,
        "abc": 5,
        "<0x64>": 6,
        "<0xC3>": 7,
        "<0x81>": 8,
    }
    merges = {("a", "b"): 1, ("ab", "c"): 2}
    tokenizer = BPETokenizer(vocab, merges, byte_fallback=True)
    ids = tokenizer.encode("d")
    assert ids == [6]
    ids2 = tokenizer.encode("Á")
    assert ids2 == [7, 8]
    assert tokenizer.encode("e") == [0]
    tokenizer_no_fb = BPETokenizer(vocab, merges, byte_fallback=False)
    assert tokenizer_no_fb.encode("d") == [0]
    assert tokenizer.decode([99]) == "[UNK]"
    tokenizer.cache = {}
    assert tokenizer.bpe("a") == ["a"]
    assert tokenizer.bpe("xy") == ["x", "y"]


def test_bpe_empty_string():
    """Provides semantic functionality and verification."""
    vocab = {"[UNK]": 0}
    merges = {}
    tokenizer = BPETokenizer(vocab, merges)
    assert tokenizer.bpe("") == []


def test_bpe_value_error_branch():
    """Provides semantic functionality and verification."""
    vocab = {"[UNK]": 0, "a": 1, "b": 2}
    merges = {("x", "y"): 1}
    tokenizer = BPETokenizer(vocab, merges)
    tokenizer.merges = {("a", "c"): 1}
    call_count = [0]

    def mock_get_pairs(w):
        """Provides semantic functionality and verification."""
        if call_count[0] == 0:
            call_count[0] += 1
            return set([("a", "c")])
        return set()

    tokenizer._get_pairs = mock_get_pairs
    assert tokenizer.bpe("ab") == ["a", "b"]
    tokenizer2 = BPETokenizer({"[UNK]": 0, "ac": 1}, {("a", "c"): 1})
    assert tokenizer2.bpe("ac") == ["ac"]


def test_tokenizer_normalization():
    """Provides semantic functionality and verification."""
    vocab = {"[UNK]": 0, "a": 1}
    tokenizer = BPETokenizer(vocab, {})
    text = "Á"
    norm = tokenizer.normalize(text, lower=True, strip_accents=True)
    assert norm == "a"


def test_tokenizer_truncation():
    """Provides semantic functionality and verification."""
    vocab = {"[UNK]": 0, "[SEP]": 1, "a": 2, "b": 3}
    tokenizer = WordPieceTokenizer(vocab, sep_token="[SEP]")
    out = tokenizer.encode_plus("a a a a", max_length=3, truncation=True)
    assert out["input_ids"] == [2, 2, 1]


def test_wordpiece_tokenizer_basic():
    """Provides semantic functionality and verification."""
    vocab = {"[UNK]": 0, "hello": 1, "world": 2, "##s": 3, "a": 4}
    tokenizer = WordPieceTokenizer(vocab, max_input_chars_per_word=10)
    ids = tokenizer.encode("hello worlds")
    assert ids == [1, 2, 3]
    assert tokenizer.decode([1, 2, 3]) == "hello worlds"
    tokenizer.max_input_chars_per_word = 2
    assert tokenizer.encode("hello") == [0]
    tokenizer.max_input_chars_per_word = 10
    assert tokenizer.encode("unknown") == [0]
    assert tokenizer.decode([99]) == "[UNK]"


def test_encode_plus():
    """Provides semantic functionality and verification."""
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "a": 4, "b": 5}
    tokenizer = WordPieceTokenizer(
        vocab, pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]"
    )
    out = tokenizer.encode_plus("a", "b", max_length=5, pad_to_max_length=True)
    assert out["input_ids"] == [1, 4, 2, 5, 2]
    out2 = tokenizer.encode_plus("a", max_length=5, pad_to_max_length=True)
    assert out2["input_ids"] == [1, 4, 2, 3, 3]


def test_from_huggingface_mock(tmpdir):
    """Provides semantic functionality and verification."""
    data = {
        "model": {"vocab": {"a": 1, "b": 2}, "merges": ["a b"]},
        "added_tokens": [{"id": 0, "content": "[UNK]"}, {"id": 3, "content": "[MASK]"}],
    }
    file = tmpdir.join("tokenizer.json")
    with open(file, "w") as f:
        json.dump(data, f)
    tokenizer = BPETokenizer.from_huggingface(str(file))
    assert tokenizer.vocab["a"] == 1
    assert tokenizer.vocab["[MASK]"] == 3
    assert tokenizer.merges["a", "b"] == 0
    tokenizer2 = BPETokenizer.from_huggingface(data)
    assert tokenizer2.vocab["a"] == 1


def test_abstract_class_fails():
    """Provides semantic functionality and verification."""
    with pytest.raises(TypeError):
        Tokenizer({}, "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]")


class DummyTokenizer(Tokenizer):
    """Provides semantic functionality and verification."""

    def encode(self, text):
        """Provides semantic functionality and verification."""
        return super().encode(text)

    def decode(self, ids):
        """Provides semantic functionality and verification."""
        return super().decode(ids)


def test_dummy_tokenizer():
    """Provides semantic functionality and verification."""
    dt = DummyTokenizer({})
    assert dt.encode("test") == []
    assert dt.decode([1]) == ""
