"""Tests for tokenizer."""

from onnx9000.onnx2gguf.tokenizer import extract_tokenizer_metadata


def test_extract_tokenizer():
    """Provides functional implementation."""
    meta = extract_tokenizer_metadata()
    assert meta["tokenizer.ggml.model"] == "llama"
    assert len(meta["tokenizer.ggml.tokens"]) == 2

    meta2 = extract_tokenizer_metadata(None, 5)
    assert len(meta2["tokenizer.ggml.tokens"]) == 5

    json_str = '{"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": ["a b"]}, "added_tokens": [{"id": 0}], "chat_template": "hello"}'
    meta3 = extract_tokenizer_metadata(json_str)
    assert meta3["tokenizer.ggml.model"] == "gpt2"
    assert meta3["tokenizer.ggml.tokens"] == ["a", "b"]
    assert meta3["tokenizer.ggml.merges"] == ["a b"]
    assert meta3["tokenizer.chat_template"] == "hello"

    meta4 = extract_tokenizer_metadata(json_str, 5)
    assert len(meta4["tokenizer.ggml.tokens"]) == 5

    meta5 = extract_tokenizer_metadata(json_str, 1)
    assert len(meta5["tokenizer.ggml.tokens"]) == 1

    json_str_unigram = '{"model": {"type": "Unigram", "vocab": {"a": 0, "b": 1}}}'
    meta6 = extract_tokenizer_metadata(json_str_unigram)
    assert meta6["tokenizer.ggml.model"] == "llama"

    invalid_json = "{invalid"
    meta7 = extract_tokenizer_metadata(invalid_json)
    assert meta7["tokenizer.ggml.model"] == "llama"

    json_str_other = '{"model": {"type": "WordPiece", "vocab": {"a": 0, "b": 1}}}'
    meta8 = extract_tokenizer_metadata(json_str_other)
    assert meta8["tokenizer.ggml.model"] == "llama"
