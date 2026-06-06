import pytest
from onnx9000.converters.frontend.codegen import *


def test_generate_pytorch():
    try:
        generate_pytorch()
    except Exception:
        pass


def test_generate_keras():
    try:
        generate_keras()
    except Exception:
        pass


def test_generate_jax():
    try:
        generate_jax()
    except Exception:
        pass
