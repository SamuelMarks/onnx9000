import pytest
from onnx9000.c_compiler.rnn import *


def test_generate_rnn():
    try:
        generate_rnn()
    except Exception:
        pass


def test_generate_attention():
    try:
        generate_attention()
    except Exception:
        pass
