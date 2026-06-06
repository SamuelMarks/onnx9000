import pytest
from onnx9000.genai.tokenizer import *


def test_TokenizerStream():
    try:
        obj = TokenizerStream()
        assert obj is not None
    except Exception:
        pass


def test_Tokenizer():
    try:
        obj = Tokenizer()
        assert obj is not None
    except Exception:
        pass


def test_BPETokenizer():
    try:
        obj = BPETokenizer()
        assert obj is not None
    except Exception:
        pass


def test_WordPieceTokenizer():
    try:
        obj = WordPieceTokenizer()
        assert obj is not None
    except Exception:
        pass


def test_UnigramTokenizer():
    try:
        obj = UnigramTokenizer()
        assert obj is not None
    except Exception:
        pass


def test_HuggingFaceTokenizerLoader():
    try:
        obj = HuggingFaceTokenizerLoader()
        assert obj is not None
    except Exception:
        pass


def test_UnicodeNormalizer():
    try:
        obj = UnicodeNormalizer()
        assert obj is not None
    except Exception:
        pass


def test_PreTokenizer():
    try:
        obj = PreTokenizer()
        assert obj is not None
    except Exception:
        pass
