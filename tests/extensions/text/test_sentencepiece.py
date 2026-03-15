"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.extensions.text.sentencepiece import (
    SentencePieceTokenizer,
    spm_normalize,
    spm_pre_tokenize,
    spm_decode,
)
import struct


def test_spm_utils():
    """Provides semantic functionality and verification."""
    assert spm_normalize("ﬃ") == "ffi"
    assert spm_pre_tokenize("hello world") == " hello world"
    assert spm_decode([" hello", " world"]) == "hello world"
    assert spm_decode(["hello", " world"]) == "hello world"


def test_sentencepiece_tokenizer():
    """Provides semantic functionality and verification."""
    vocab = {
        "[UNK]": 0,
        " a": 1,
        " b": 2,
        "c": 3,
        "<0xE2>": 4,
        "<0x82>": 5,
        "<0xAC>": 6,
        " a b": 7,
    }
    scores = {k: (-1.0) for k in vocab}
    tokenizer = SentencePieceTokenizer(vocab, scores, byte_fallback=True)
    ids = tokenizer.encode("a b")
    assert ids == [7]
    assert tokenizer.decode([1, 2]) == "a b"
    euro_ids = [4, 5, 6]
    assert tokenizer.decode(euro_ids) == "€"
    bad_byte = [4, 99]
    decoded = tokenizer.decode(bad_byte)
    assert decoded == "�[UNK]"


def test_spm_file_load(tmpdir):
    """Provides semantic functionality and verification."""
    spm_path = tmpdir.join("test.model")

    def make_piece(s: str, score: float):
        """Provides semantic functionality and verification."""
        score_bytes = struct.pack("<f", score)
        inner = (
            bytes([10, len(s.encode("utf-8"))])
            + s.encode("utf-8")
            + bytes([21])
            + score_bytes
            + bytes([24, 1])
        )
        return bytes([10, len(inner)]) + inner

    buf = make_piece("[UNK]", 0.0) + make_piece(" a", -1.0)
    with open(spm_path, "wb") as f:
        f.write(buf)
    tokenizer = SentencePieceTokenizer.from_spm_file(str(spm_path), byte_fallback=True)
    assert tokenizer.vocab[" a"] == 1
    assert tokenizer.byte_fallback is True


def test_spm_bad_byte_in_token():
    """Provides semantic functionality and verification."""
    vocab = {"[UNK]": 0, "<0xZZ>": 1}
    scores = {"[UNK]": 0.0, "<0xZZ>": -1.0}
    tokenizer = SentencePieceTokenizer(vocab, scores, byte_fallback=True)
    assert tokenizer.decode([1]) == "<0xZZ>"


def test_spm_pre_tokenize_coverage():
    """Provides semantic functionality and verification."""
    from onnx9000.extensions.text.sentencepiece import SentencePieceTokenizer

    tok = SentencePieceTokenizer({"a": 1}, {"a": 1.0})
    assert tok.pre_tokenize("abc") == ["abc"]
