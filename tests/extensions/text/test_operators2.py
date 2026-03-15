"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.extensions.text.operators import (
    tf_idf_vectorizer,
    wordpiece_tokenizer,
    n_gram_extraction,
)


def test_tf_idf_vectorizer():
    """Provides semantic functionality and verification."""
    x = [["hello", "world"], ["hello", "hello"]]
    pool = ["hello", "world"]
    ngram_counts = [1, 1]
    ngram_indexes = [0, 1]
    weights = [1.0, 2.0]
    out = tf_idf_vectorizer(
        x=x,
        min_gram_length=1,
        max_gram_length=1,
        max_skip_count=0,
        mode="TF",
        ngram_counts=ngram_counts,
        ngram_indexes=ngram_indexes,
        pool_strings=pool,
        weights=weights,
    )
    assert out[0] == [1.0, 1.0]
    assert out[1] == [2.0, 0.0]
    out_idf = tf_idf_vectorizer(
        x=x,
        min_gram_length=1,
        max_gram_length=1,
        max_skip_count=0,
        mode="IDF",
        ngram_counts=ngram_counts,
        ngram_indexes=ngram_indexes,
        pool_strings=pool,
        weights=weights,
    )
    assert out_idf[0] == [1.0, 2.0]
    assert out_idf[1] == [1.0, 0.0]
    out_tfidf = tf_idf_vectorizer(
        x=x,
        min_gram_length=1,
        max_gram_length=1,
        max_skip_count=0,
        mode="TFIDF",
        ngram_counts=ngram_counts,
        ngram_indexes=ngram_indexes,
        pool_strings=pool,
        weights=weights,
    )
    assert out_tfidf[0] == [1.0, 2.0]
    assert out_tfidf[1] == [2.0, 0.0]


def test_tf_idf_no_pool():
    """Provides semantic functionality and verification."""
    out = tf_idf_vectorizer([["hi"]], 1, 1, 0, "TF", [], [])
    assert out == [[]]


def test_wordpiece_tokenizer_op():
    """Provides semantic functionality and verification."""
    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    out = wordpiece_tokenizer(["hello", "world", "unknown"], vocab)
    assert out == [[1], [2], [0]]


def test_n_gram_extraction():
    """Provides semantic functionality and verification."""
    tokens = ["a", "b", "c", "d"]
    assert n_gram_extraction(tokens, 2) == ["a b", "b c", "c d"]
    assert n_gram_extraction(tokens, 0) == []
    assert n_gram_extraction([], 1) == []
