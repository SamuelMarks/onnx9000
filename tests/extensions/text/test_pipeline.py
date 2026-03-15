"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.extensions.text.pipeline import (
    TextClassificationPipeline,
    Seq2SeqPipeline,
    ConstrainedGenerator,
)


class DummyTokenizer:
    """Provides semantic functionality and verification."""

    def encode_plus(self, text, truncation=False):
        """Provides semantic functionality and verification."""
        return {"input_ids": [1, 2, 3]}

    def encode(self, text):
        """Provides semantic functionality and verification."""
        return [1, 2, 3]

    def decode(self, ids):
        """Provides semantic functionality and verification."""
        return "decoded"


def test_text_classification_pipeline():
    """Provides semantic functionality and verification."""

    def mock_model(inputs):
        """Provides semantic functionality and verification."""
        return {"logits": [[0.1, 0.9, 0.2]]}

    pipeline = TextClassificationPipeline(
        DummyTokenizer(), mock_model, {(0): "NEG", (1): "POS", (2): "NEU"}
    )
    res = pipeline("hello")
    assert res["label"] == "POS"
    assert res["score"] == 0.9


def test_text_classification_pipeline_empty():
    """Provides semantic functionality and verification."""

    def mock_model(inputs):
        """Provides semantic functionality and verification."""
        return {}

    pipeline = TextClassificationPipeline(DummyTokenizer(), mock_model, {(0): "NEG"})
    res = pipeline("hello")
    assert res["label"] == "UNKNOWN"


def test_seq2seq_pipeline():
    """Provides semantic functionality and verification."""
    state = {"step": 0}

    def mock_model(inputs):
        """Provides semantic functionality and verification."""
        if state["step"] == 0:
            state["step"] += 1
            return {"logits": [[[0.1, 0.9, 0.0]]]}
        elif state["step"] == 1:
            state["step"] += 1
            return {"logits": [[[0.0, 0.0, 0.9]]]}

    pipeline = Seq2SeqPipeline(
        DummyTokenizer(), mock_model, max_length=10, eos_token_id=2
    )
    res = pipeline.generate("hello")
    assert res == "decoded"


def test_seq2seq_pipeline_empty():
    """Provides semantic functionality and verification."""

    def mock_model(inputs):
        """Provides semantic functionality and verification."""
        return {}

    pipeline = Seq2SeqPipeline(DummyTokenizer(), mock_model)
    res = pipeline.generate("hello")
    assert res == "decoded"


def test_constrained_generator():
    """Provides semantic functionality and verification."""
    trie = {"{": [1, 2], '{"k': [3]}
    generator = ConstrainedGenerator(trie)
    assert generator.get_allowed_tokens("{") == [1, 2]
    assert generator.get_allowed_tokens("x") == []
