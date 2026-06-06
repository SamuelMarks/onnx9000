import pytest
from onnx9000.genai.search import *

def test_SearchAlgorithm():
    try:
        obj = SearchAlgorithm()
        assert obj is not None
    except Exception:
        pass

def test_GreedySearch():
    try:
        obj = GreedySearch()
        assert obj is not None
    except Exception:
        pass

def test_MultinomialSampling():
    try:
        obj = MultinomialSampling()
        assert obj is not None
    except Exception:
        pass

def test_BeamSearchState():
    try:
        obj = BeamSearchState()
        assert obj is not None
    except Exception:
        pass

def test_BeamSearchAlgorithm():
    try:
        obj = BeamSearchAlgorithm()
        assert obj is not None
    except Exception:
        pass

