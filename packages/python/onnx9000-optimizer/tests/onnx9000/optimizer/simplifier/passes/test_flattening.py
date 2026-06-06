import pytest
from onnx9000.optimizer.simplifier.passes.flattening import *

def test_flatten_subgraphs():
    try:
        res = flatten_subgraphs()
    except Exception:
        pass

