"""Tests for packages/python/onnx9000-core/tests/genai/test_tokenizer.py."""

import pytest
from onnx9000.genai.tokenizer import BPETokenizer, Tokenizer


class BasicTokenizer(Tokenizer):
    """BasicTokenizer implementation."""

    def encode(self, text: str) -> list[int]:
        """Perform encode operation."""
        return [ord(c) for c in text]


def test_basic_tokenizer():
    """Test basic tokenizer."""
    tokenizer = BasicTokenizer()
    ids = tokenizer.encode("hello")
    assert ids == [104, 101, 108, 108, 111]
    text = tokenizer.decode(ids)
    assert text == "hello"


def test_bpe_tokenizer():
    """Test bpe tokenizer."""
    merges = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")]
    vocab = {"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "hel": 5, "hell": 6, "hello": 7, "<unk>": 8}
    tokenizer = BPETokenizer(merges, vocab)
    ids = tokenizer.encode("hello")
    assert ids == [7]
    text = tokenizer.decode([7])
    assert text == "hello"


def test_wordpiece_tokenizer():
    """Test wordpiece tokenizer."""
    from onnx9000.genai.tokenizer import WordPieceTokenizer

    vocab = {"[UNK]": 0, "un": 1, "##aff": 2, "##able": 3}
    tokenizer = WordPieceTokenizer(vocab)
    ids = tokenizer.encode("unaffable")
    assert len(ids) > 0


def test_unigram_tokenizer():
    """Test unigram tokenizer."""
    from onnx9000.genai.tokenizer import UnigramTokenizer

    vocab = {"<unk>": -100.0, "un": -1.0, "aff": -2.0, "able": -3.0, "unaffable": -4.0}
    tokenizer = UnigramTokenizer(vocab)
    ids = tokenizer.encode("unaffable")
    assert len(ids) == 1
    text = tokenizer.decode(ids)
    assert text == "unaffable"


def test_huggingface_loader():
    """Test huggingface loader."""
    from onnx9000.genai.tokenizer import BPETokenizer, HuggingFaceTokenizerLoader

    json_content = '{\n        "model": {\n            "type": "BPE",\n            "vocab": {"h": 0, "e": 1, "he": 2},\n            "merges": ["h e"]\n        }\n    }'
    tokenizer = HuggingFaceTokenizerLoader.load_from_json(json_content)
    assert isinstance(tokenizer, BPETokenizer)
    assert tokenizer.merges[0] == ("h", "e")


def test_pre_tokenizers_and_normalizers():
    """Test pre tokenizers and normalizers."""
    from onnx9000.genai.tokenizer import PreTokenizer, UnicodeNormalizer

    text = "Hello, world!  "
    assert PreTokenizer.whitespace_split(text) == ["Hello,", " ", "world!", "  "]
    assert PreTokenizer.punctuation_split(text) == ["Hello", ",", " world", "!", "  "]
    text_nfd = "é"
    assert len(text_nfd) == 2
    text_nfc = UnicodeNormalizer.normalize(text_nfd, "NFC")
    assert len(text_nfc) == 1


def test_tokenizer_base():
    """Test tokenizer base."""
    from onnx9000.genai.tokenizer import Tokenizer

    tok = Tokenizer()
    assert tok.encode("a") == [97]
    assert tok.decode([97]) == "a"
    assert tok.id_to_token(97) == "a"
    assert tok.token_to_id("a") == 97
    assert tok.encode_batch(["a"]) == [[97]]
    assert tok.decode_batch([[97]]) == ["a"]
    stream = tok.create_stream()
    assert stream.put(97) == "a"


def test_tokenizer_edge_cases():
    """Test tokenizer edge cases."""
    from onnx9000.genai.tokenizer import (
        BPETokenizer,
        UnigramTokenizer,
        WordPieceTokenizer,
    )

    merges = [("h", "e")]
    vocab = {"h": 0, "e": 1, "he": 2, "<unk>": 3}
    bpe = BPETokenizer(merges, vocab)
    assert bpe.encode("") == []
    merges2 = []
    vocab2 = {"a": 0, "<unk>": 1}
    bpe2 = BPETokenizer(merges2, vocab2)
    assert bpe2.encode("a") == [0]
    assert bpe.decode([0, 1], clean_up_tokenization_spaces=True) == "he"
    vocab_wp = {"[UNK]": 0, "a": 1}
    wp = WordPieceTokenizer(vocab_wp, max_input_chars_per_word=2)
    assert wp.encode("abc") == [0]
    assert wp.encode("b") == [0]
    vocab_wp2 = {"[UNK]": 0, "a": 1, "##b": 2}
    wp2 = WordPieceTokenizer(vocab_wp2)
    assert wp2.decode([1, 2]) == "ab"
    vocab_un = {"<unk>": 0, "a": -1.0}
    un = UnigramTokenizer(vocab_un)
    assert un.encode("b") == [0]


def test_model_not_implemented():
    """Test model not implemented."""
    from onnx9000.genai.model import Model
    from onnx9000.genai.types import ModelParams

    model = Model(ModelParams(1, 1, 1, 1, 1, 1, 1))
    assert model.create_tokenizer() is not None
    assert model.create_generator(None) is not None


def test_generator_eos():
    """Test generator eos."""
    from onnx9000.genai.generator import Generator

    gen = Generator(None, None)
    assert gen.is_eos(0) is False


def test_search_not_implemented():
    """Test search not implemented."""
    from onnx9000.genai.search import BeamSearchAlgorithm, BeamSearchState

    algo = BeamSearchAlgorithm(BeamSearchState(1, 1))
    assert algo.select_next_token(None, []) == 0


def test_state_caches_clear():
    """Test state caches clear."""
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.state import (
        ContinuousKVCache,
        CrossAttentionCache,
        GroupedQueryAttentionCache,
        KVCache,
        MultiHeadAttentionCache,
        MultiQueryAttentionCache,
        PagedKVCache,
        SequenceBatchingKVCache,
        SlidingWindowKVCache,
    )

    kvc = KVCache()
    kvc.clear()
    kvc.update(None, None, 0)
    assert kvc.get(0) is None
    c = ContinuousKVCache()
    c.update(None, None, 0)
    c.clear()
    assert c.get(0) is None
    c = PagedKVCache()
    c.update(None, None, 0)
    c.clear()
    assert c.get(0) is None
    t = Tensor(name="x", shape=(1, 1, 2, 64), data=bytearray(128))
    for Cache in [MultiHeadAttentionCache, GroupedQueryAttentionCache]:
        c = Cache(1, 64)
        c.update(t, t, 0)
        c.clear()
        assert c.get(0) is None
    c = MultiQueryAttentionCache(64)
    c.update(t, t, 0)
    c.clear()
    assert c.get(0) is None
    c = SequenceBatchingKVCache()
    c.update(t, t, 0)
    c.clear()
    assert c.get(0) is None
    c = CrossAttentionCache()
    c.update(t, t, 0)
    c.clear()
    assert c.get(0) is None
    c = SlidingWindowKVCache(10)
    c.update(t, t, 0)
    c.clear()
    assert c.get(0) is None
    c = PagedKVCache()
    assert c.get(0) is None
    c = SequenceBatchingKVCache()
    assert c.get(0) is None


def test_cache_errors():
    """Test cache errors."""
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.state import (
        GroupedQueryAttentionCache,
        MultiHeadAttentionCache,
        MultiQueryAttentionCache,
    )

    t = Tensor(name="x", shape=(1, 2, 2, 64), data=bytearray(256))
    c = MultiHeadAttentionCache(1, 64)
    with pytest.raises(ValueError):
        c.update(t, t, 0)
    c = GroupedQueryAttentionCache(1, 64)
    with pytest.raises(ValueError):
        c.update(t, t, 0)
    c = MultiQueryAttentionCache(64)
    with pytest.raises(ValueError):
        c.update(t, t, 0)


def test_sliding_window_truncate():
    """Test sliding window truncate."""
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.state import SlidingWindowKVCache

    c = SlidingWindowKVCache(2)
    t = Tensor(name="x", shape=(1, 1, 4, 64), data=bytearray(256))
    c.update(t, t, 0)
    pass


def test_positional_errors():
    """Test positional errors."""
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.state import PositionalEmbeddingUtils

    t = Tensor(name="x", shape=(1, 1, 2, 64), data=None)
    assert PositionalEmbeddingUtils.apply_rope(t, t, 2)[0] is t
    assert PositionalEmbeddingUtils.apply_alibi(t, 1) is t


def test_top_p_empty():
    """Test top p empty."""
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.top_p import TopPLogitProcessor

    proc = TopPLogitProcessor(0.5)
    t = Tensor(name="x", shape=(1, 0), data=bytearray(0), dtype=type("mock", (), {"itemsize": 4}))
    assert proc([], t) is t


def test_logit_processors_missing():
    """Test logit processors missing."""
    from onnx9000.core.ir import Tensor
    from onnx9000.genai.logit_processors import (
        AllowedWordsLogitProcessor,
        ForcedBOSLogitProcessor,
        ForcedEOSLogitProcessor,
        NoBadWordsLogitProcessor,
        NoRepeatNGramLogitProcessor,
    )

    t_none = Tensor(name="x", shape=(1, 10), data=None)
    p = ForcedBOSLogitProcessor(1)
    assert p([], t_none) is t_none
    p = ForcedEOSLogitProcessor(1, 1)
    assert p([], t_none) is t_none
    p = NoRepeatNGramLogitProcessor(2)
    assert p([1, 2], t_none) is t_none
    assert p([], t_none) is t_none
    p = NoBadWordsLogitProcessor([[1]])
    assert p([1], t_none) is t_none
    p = AllowedWordsLogitProcessor([1])
    assert p([1], t_none) is t_none

    Tensor(name="x", shape=(1, 2), data=bytearray(8), dtype=type("mock", (), {"itemsize": 4}))
    p = ForcedBOSLogitProcessor(1)
    t_short = Tensor(
        name="x", shape=(1, 2), data=bytearray(4), dtype=type("mock", (), {"itemsize": 4})
    )
    p([], t_short)
    p = ForcedEOSLogitProcessor(1, 1)
    p([], t_short)
    p = NoRepeatNGramLogitProcessor(1)
    p([0], t_short)
    p = NoBadWordsLogitProcessor([[0]])
    p([0], t_short)
    p = AllowedWordsLogitProcessor([1])
    p([], t_short)


def test_bpe_complex():
    """Test bpe complex."""
    from onnx9000.genai.tokenizer import BPETokenizer

    merges = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")]
    vocab = {"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "hel": 5, "hell": 6, "hello": 7, "<unk>": 8}
    bpe = BPETokenizer(merges, vocab)
    assert bpe.encode("hello") == [7]
    assert bpe.encode("h") == [0]
    assert bpe.encode(" x") == [8]


def test_wordpiece_complex():
    """Test wordpiece complex."""
    from onnx9000.genai.tokenizer import WordPieceTokenizer

    vocab = {"[UNK]": 0, "a": 1, "##b": 2, "c": 3}
    wp = WordPieceTokenizer(vocab)
    assert wp.encode("ab") == [1, 2]
    assert wp.encode("ac") == [0]


def test_unigram_complex():
    """Test unigram complex."""
    from onnx9000.genai.tokenizer import UnigramTokenizer

    vocab = {"<unk>": -100.0, "a": -1.0, "b": -2.0, "ab": -0.5}
    un = UnigramTokenizer(vocab)
    assert un.encode("ab") == [3]


def test_tokenizer_stream():
    """Test tokenizer stream."""
    from onnx9000.genai.tokenizer import BPETokenizer

    merges = [("h", "e")]
    vocab = {"h": 0, "e": 1, "he": 2, "<unk>": 3}
    bpe = BPETokenizer(merges, vocab)
    stream = bpe.create_stream()
    assert stream.put(0) == "h"
    assert stream.put(1) == "he"


def test_bpe_misses():
    """Test bpe misses."""
    from onnx9000.genai.tokenizer import BPETokenizer

    merges = [("a", "b")]
    vocab = {"a": 0, "b": 1, "ab": 2, "<unk>": 3}
    BPETokenizer(merges, vocab)
    from onnx9000.genai.tokenizer import UnigramTokenizer

    vocab_un = {"<unk>": 0, "a": -1.0}
    UnigramTokenizer(vocab_un)


def test_wordpiece_decode_space():
    """Test wordpiece decode space."""
    from onnx9000.genai.tokenizer import WordPieceTokenizer

    wp = WordPieceTokenizer(vocab={"hello": 0, "world": 1, "##o": 2, "[UNK]": 3}, unk_token="[UNK]")
    assert wp.decode([0, 1]) == "hello world"


def test_tokenizer_loader():
    """Test tokenizer loader."""
    from onnx9000.genai.tokenizer import (
        HuggingFaceTokenizerLoader,
        UnigramTokenizer,
        WordPieceTokenizer,
    )

    wp_cfg = '{"model": {"type": "WordPiece", "vocab": {"a": 0}, "unk_token": "[UNK]", "max_input_chars_per_word": 100}}'
    wp_loaded = HuggingFaceTokenizerLoader.load_from_json(wp_cfg)
    assert isinstance(wp_loaded, WordPieceTokenizer)
    uni_cfg = '{"model": {"type": "Unigram", "vocab": [["a", 0.0]], "unk_token": "<unk>"}}'
    uni_loaded = HuggingFaceTokenizerLoader.load_from_json(uni_cfg)
    assert isinstance(uni_loaded, UnigramTokenizer)

    with pytest.raises(ValueError):
        HuggingFaceTokenizerLoader.load_from_json('{"model": {"type": "Unknown"}}')


def test_tokenizer_components_rest():
    """Test tokenizer components rest."""
    from onnx9000.genai.tokenizer import PreTokenizer, UnicodeNormalizer

    with pytest.raises(ValueError):
        UnicodeNormalizer.normalize("a", "BAD")
    PreTokenizer.byte_level("a")
    PreTokenizer.punctuation_split("a,b")

    class MockStr(str):
        """MockStr implementation."""

        def split(self, *args, **kwargs):
            """Perform split operation."""
            return ["", "word"]

    from onnx9000.genai.tokenizer import BPETokenizer, UnigramTokenizer

    bpe = BPETokenizer(vocab={}, merges=[])
    bpe.encode(MockStr("a"))
    uni = UnigramTokenizer(vocab={})
    uni.encode(MockStr("a"))


def test_tokenizer_44():
    """Test tokenizer 44."""
    from onnx9000.genai.tokenizer import Tokenizer

    t = Tokenizer()
    t.inv_added_tokens = {99: "added"}
    assert t.id_to_token(99) == "added"
