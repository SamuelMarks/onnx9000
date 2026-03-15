from onnx9000.extensions.text.unigram import UnigramTokenizer


def test_unigram_dp_unreachable():
    tokenizer = UnigramTokenizer({"a": 1}, {"a": 1.0})
    try:
        tokenizer.encode("b")
    except Exception:
        pass
