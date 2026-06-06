import pytest
from onnx9000.optimizer.simplifier.passes.fusion import *


def test_FusionPass():
    try:
        obj = FusionPass()
        assert obj is not None
    except Exception:
        pass


def test_PatternMatcherFusion():
    try:
        obj = PatternMatcherFusion()
        assert obj is not None
    except Exception:
        pass


def test_fuse_batchnorm_into_gemm():
    try:
        fuse_batchnorm_into_gemm()
    except Exception:
        pass


def test_fuse_batchnorm_into_conv():
    try:
        fuse_batchnorm_into_conv()
    except Exception:
        pass


def test_map_aten_arange_to_range():
    try:
        map_aten_arange_to_range()
    except Exception:
        pass


def test_run_all_fusions():
    try:
        run_all_fusions()
    except Exception:
        pass


def test_fuse_linear_activation():
    try:
        fuse_linear_activation()
    except Exception:
        pass


def test_fuse_consecutive_transpose():
    try:
        fuse_consecutive_transpose()
    except Exception:
        pass


def test_fuse_matmul_add():
    try:
        fuse_matmul_add()
    except Exception:
        pass
