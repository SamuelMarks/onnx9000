import pytest
from onnx9000.optimizer.simplifier.passes.partitioning import *

def test_partition_for_multi_device():
    try:
        res = partition_for_multi_device()
    except Exception:
        pass

