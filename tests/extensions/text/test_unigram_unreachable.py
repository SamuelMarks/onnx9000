import pytest
from onnx9000.extensions.text.unigram import UnigramTokenizer


def test_unigram_unreachable():
    vocab = {"a": 0, "b": 1, "<unk>": 2}
    scores = {"a": 1.0, "b": 2.0}
    tokenizer = UnigramTokenizer(vocab, scores=scores, unk_token="<unk>")
    tokenizer.unk_score = -float("inf")
    res = tokenizer.encode("cd")
    assert isinstance(res, list)
