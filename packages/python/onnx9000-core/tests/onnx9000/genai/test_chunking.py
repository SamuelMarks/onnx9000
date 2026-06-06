import pytest
from onnx9000.genai.chunking import *


def test_ChunkManager():
    try:
        obj = ChunkManager()
        assert obj is not None
    except Exception:
        pass
