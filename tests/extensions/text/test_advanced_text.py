"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.extensions.text.tokenizer import BPETokenizer


def test_zero_width_and_emojis():
    """Provides semantic functionality and verification."""
    vocab = {"[UNK]": 0, "a": 1, "👨": 2, "💻": 3, "\u200d": 4}
    merges = {("👨", "\u200d"): 1, ("👨\u200d", "💻"): 2}
    tokenizer = BPETokenizer(vocab, merges)
    ids = tokenizer.encode("👨\u200d💻")
    pass
