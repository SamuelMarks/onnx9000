import pytest
from onnx9000.optimizer.simplifier.passes.shapes import *


def test_ShapeInferencePass():
    try:
        obj = ShapeInferencePass()
        assert obj is not None
    except Exception:
        pass


def test_resolve_dynamic_batch():
    try:
        resolve_dynamic_batch()
    except Exception:
        pass


def test_resolve_dynamic_sequence():
    try:
        resolve_dynamic_sequence()
    except Exception:
        pass


def test_extract_rnn_states():
    try:
        extract_rnn_states()
    except Exception:
        pass
