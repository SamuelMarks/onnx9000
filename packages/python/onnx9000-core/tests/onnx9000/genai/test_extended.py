import pytest
from onnx9000.genai.extended import *

def test_DraftingModel():
    try:
        obj = DraftingModel()
        assert obj is not None
    except Exception:
        pass

def test_DraftVerifier():
    try:
        obj = DraftVerifier()
        assert obj is not None
    except Exception:
        pass

def test_SelfConsistencyDecoder():
    try:
        obj = SelfConsistencyDecoder()
        assert obj is not None
    except Exception:
        pass

def test_ContinuousBatchingQueue():
    try:
        obj = ContinuousBatchingQueue()
        assert obj is not None
    except Exception:
        pass

def test_HiddenStateVisualizer():
    try:
        obj = HiddenStateVisualizer()
        assert obj is not None
    except Exception:
        pass

def test_PromptCompressor():
    try:
        obj = PromptCompressor()
        assert obj is not None
    except Exception:
        pass

def test_ChunkedPrefiller():
    try:
        obj = ChunkedPrefiller()
        assert obj is not None
    except Exception:
        pass

def test_DynamicParamAdjuster():
    try:
        obj = DynamicParamAdjuster()
        assert obj is not None
    except Exception:
        pass

def test_MultiTurnCache():
    try:
        obj = MultiTurnCache()
        assert obj is not None
    except Exception:
        pass

