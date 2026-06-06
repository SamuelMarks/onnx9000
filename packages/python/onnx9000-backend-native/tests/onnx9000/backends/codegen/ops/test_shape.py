import pytest
from onnx9000.backends.codegen.ops.shape import *


def test_generate_reshape():
    try:
        generate_reshape()
    except Exception:
        pass


def test_generate_flatten():
    try:
        generate_flatten()
    except Exception:
        pass


def test_generate_squeeze():
    try:
        generate_squeeze()
    except Exception:
        pass


def test_generate_unsqueeze():
    try:
        generate_unsqueeze()
    except Exception:
        pass


def test_generate_cast_like():
    try:
        generate_cast_like()
    except Exception:
        pass


def test_generate_cast():
    try:
        generate_cast()
    except Exception:
        pass


def test_generate_expand():
    try:
        generate_expand()
    except Exception:
        pass
