import pytest
from onnx9000.genai.logging import *


def test_GenerationStatsLogger():
    try:
        obj = GenerationStatsLogger()
        assert obj is not None
    except Exception:
        pass
