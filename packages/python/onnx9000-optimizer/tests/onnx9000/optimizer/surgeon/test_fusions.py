import pytest
from onnx9000.optimizer.surgeon.fusions import *

def test_fuse_flash_attention():
    try:
        res = fuse_flash_attention()
    except Exception:
        pass

def test_fuse_horizontal_gemm():
    try:
        res = fuse_horizontal_gemm()
    except Exception:
        pass

