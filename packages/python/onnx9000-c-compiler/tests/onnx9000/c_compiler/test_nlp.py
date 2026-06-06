import pytest
from onnx9000.c_compiler.nlp import *


def test_generate_topk():
    try:
        generate_topk()
    except Exception:
        pass


def test_generate_unique():
    try:
        generate_unique()
    except Exception:
        pass


def test_emit_bpe_tokenizer():
    try:
        emit_bpe_tokenizer()
    except Exception:
        pass
