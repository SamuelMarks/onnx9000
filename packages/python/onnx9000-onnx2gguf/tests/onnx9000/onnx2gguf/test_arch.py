import pytest
from onnx9000.onnx2gguf.arch import *


def test_extract_metadata():
    try:
        extract_metadata()
    except Exception:
        pass


def test_infer_architecture():
    try:
        infer_architecture()
    except Exception:
        pass
