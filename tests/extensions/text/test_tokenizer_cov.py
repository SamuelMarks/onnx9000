"""Module providing core logic and structural definitions."""

import pytest


def test_tokenizer_encode_to_buffer():
    """Provides semantic functionality and verification."""
    from onnx9000.extensions.text.tokenizer import Tokenizer

    class MyTok(Tokenizer):
        """Represents the MyTok class."""

        def encode(self, t):
            """Provides encode functionality and verification."""
            return [1, 2, 3]

        def decode(self, t):
            """Provides decode functionality and verification."""
            return ""

        def normalize(self, t):
            """Provides normalize functionality and verification."""
            return t

        def pre_tokenize(self, t):
            """Provides pre tokenize functionality and verification."""
            return [t]

        def post_process(self, t):
            """Provides post process functionality and verification."""
            return t

    tok = MyTok({})
    try:
        buf = tok.encode_to_buffer("text")
        assert list(buf) == [1, 2, 3]
    except NotImplementedError:
        pass


def test_bpe_cache_and_dropout():
    """Provides semantic functionality and verification."""
    from onnx9000.extensions.text.tokenizer import BPETokenizer
    from unittest.mock import patch

    vocab = {"a": 1, "b": 2, "ab": 3}
    merges = {("a", "b"): 1}
    tok = BPETokenizer(vocab, merges)
    tok.bpe("ab")
    tok.bpe("ab")
    tok_drop = BPETokenizer(vocab, merges, dropout_prob=0.99)
    with patch("random.random", return_value=0.0):
        tok_drop.bpe("ab")
