"""Unit tests for tokenizers in `onnx9000.genai.tokenizer`."""

from onnx9000.genai.tokenizer import (
    BPETokenizer,
    HuggingFaceTokenizerLoader,
    PreTokenizer,
    Tokenizer,
    TokenizerStream,
    UnicodeNormalizer,
    UnigramTokenizer,
    WordPieceTokenizer,
)


def test_tokenizers() -> None:
    """Verify Tokenizer, TokenizerStream, BPETokenizer, WordPieceTokenizer, UnigramTokenizer, HuggingFaceTokenizerLoader, UnicodeNormalizer, and PreTokenizer."""
    tok = Tokenizer()
    ids = tok.encode("hi")
    assert ids == [104, 105]
    assert tok.decode(ids) == "hi"

    stream = TokenizerStream(tok)
    assert stream.put(104) == "h"

    bpe = BPETokenizer(merges=[("h", "i")], vocab={"h": 0, "i": 1, "hi": 2, "<unk>": 3})
    assert bpe.encode("hi") == [2]
    assert bpe.decode([2]) == "hi"

    wp = WordPieceTokenizer(vocab={"[UNK]": 0, "he": 1, "##llo": 2})
    assert wp.encode("hello") == [1, 2]

    uni = UnigramTokenizer(vocab={"<unk>": -10.0, "hello": -1.0})
    assert uni.encode("hello") == [1]

    hf_json = '{"model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}}'
    loaded = HuggingFaceTokenizerLoader.load_from_json(hf_json)
    assert isinstance(loaded, BPETokenizer)

    assert UnicodeNormalizer.normalize("abc", "NFC") == "abc"
    assert PreTokenizer.whitespace_split("a b") == ["a", " ", "b"]
