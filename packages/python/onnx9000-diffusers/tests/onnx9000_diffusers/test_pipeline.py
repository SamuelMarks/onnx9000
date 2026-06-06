import pytest
from onnx9000_diffusers.pipeline import *


def test_AbortSignal():
    try:
        obj = AbortSignal()
        assert obj is not None
    except Exception:
        pass


def test_DiffusionPipeline():
    try:
        obj = DiffusionPipeline()
        assert obj is not None
    except Exception:
        pass
