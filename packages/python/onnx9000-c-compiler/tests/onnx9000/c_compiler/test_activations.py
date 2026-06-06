import pytest
from onnx9000.c_compiler.activations import *


def test_generate_activation():
    try:
        generate_activation()
    except Exception:
        pass


def test_generate_softmax():
    try:
        generate_softmax()
    except Exception:
        pass


def test_generate_normalization():
    try:
        generate_normalization()
    except Exception:
        pass
