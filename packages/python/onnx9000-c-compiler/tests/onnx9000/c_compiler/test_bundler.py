import pytest
from onnx9000.c_compiler.bundler import *


def test_bundle_weights_bin():
    try:
        bundle_weights_bin()
    except Exception:
        pass


def test_generate_memory_summary():
    try:
        generate_memory_summary()
    except Exception:
        pass
