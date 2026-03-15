"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.extensions.custom.decoding import (
    greedy_search,
    sample_top_k_top_p,
    beam_search,
)
import random


def test_greedy_search():
    """Provides semantic functionality and verification."""
    logits = [[0.1, 0.9, 0.0], [0.2, 0.1, 0.8], [0.9, 0.0, 0.1]]
    assert greedy_search(logits) == [1, 2, 0]
    assert greedy_search(logits, eos_token_id=2) == [1, 2]


def test_sample_top_k_top_p():
    """Provides semantic functionality and verification."""
    random.seed(42)
    logits = [10.0, 10.0, 1.0, 1.0]
    res = sample_top_k_top_p(logits, top_k=2, top_p=1.0)
    assert res in [0, 1]
    res2 = sample_top_k_top_p([1.0, 2.0, 1.0], temperature=0.01)
    assert res2 == 1
    res3 = sample_top_k_top_p([10.0, 9.0, 0.0], top_k=0, top_p=0.99)
    assert res3 in [0, 1]


def test_beam_search():
    """Provides semantic functionality and verification."""

    def mock_logits(seqs):
        """Provides semantic functionality and verification."""
        out = []
        for s in seqs:
            if len(s) == 1:
                out.append([0.1, 0.9, 0.2])
            else:
                out.append([0.0, 0.0, 0.9])
        return out

    res = beam_search(mock_logits, [0], beam_size=2, max_steps=3, eos_token_id=2)
    assert res == [0, 1, 2]


def test_beam_search_max_steps():
    """Provides semantic functionality and verification."""

    def mock_logits(seqs):
        """Provides semantic functionality and verification."""
        return [[0.1, 0.9] for _ in seqs]

    res = beam_search(mock_logits, [0], beam_size=1, max_steps=2)
    assert res == [0, 1, 1]
