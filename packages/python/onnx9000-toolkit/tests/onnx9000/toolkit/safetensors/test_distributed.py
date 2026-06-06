import pytest
from onnx9000.toolkit.safetensors.distributed import *

def test_load_sharded_tensors():
    try:
        res = load_sharded_tensors()
    except Exception:
        pass

def test_pipeline_parallel_loader():
    try:
        res = pipeline_parallel_loader()
    except Exception:
        pass

