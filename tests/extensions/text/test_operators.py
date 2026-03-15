"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.extensions.text.operators import (
    string_normalizer,
    regex_replace,
    vocab_mapping,
)


def test_string_normalizer():
    """Provides semantic functionality and verification."""
    x = ["Hello", "World", "And", "hello"]
    y = string_normalizer(x, case_change_action="LOWER")
    assert y == ["hello", "world", "and", "hello"]
    y2 = string_normalizer(x, case_change_action="LOWER", stopwords=["world", "and"])
    assert y2 == ["hello", "hello"]
    y3 = string_normalizer(
        x, case_change_action="UPPER", is_case_sensitive=0, stopwords=["hello"]
    )
    assert y3 == ["WORLD", "AND"]
    y4 = string_normalizer(
        x, case_change_action="UPPER", is_case_sensitive=1, stopwords=["HELLO"]
    )
    assert y4 == ["WORLD", "AND"]


def test_regex_replace():
    """Provides semantic functionality and verification."""
    x = ["hello world", "foo bar foo"]
    y = regex_replace(x, "foo", "baz", global_replace=1)
    assert y == ["hello world", "baz bar baz"]
    y2 = regex_replace(x, "foo", "baz", global_replace=0)
    assert y2 == ["hello world", "baz bar foo"]


def test_vocab_mapping():
    """Provides semantic functionality and verification."""
    x = ["hello", "world", "unknown"]
    vocab = {"hello": 1, "world": 2}
    y = vocab_mapping(x, vocab, unk_token_id=0)
    assert y == [1, 2, 0]
