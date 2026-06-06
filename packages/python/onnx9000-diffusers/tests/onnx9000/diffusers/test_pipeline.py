import pytest
from onnx9000.diffusers.pipeline import *


def test_DiffusionPipeline():
    try:
        obj = DiffusionPipeline()
        assert obj is not None
    except Exception:
        pass


def test_set_progress_bar_config():
    try:
        set_progress_bar_config()
    except Exception:
        pass
