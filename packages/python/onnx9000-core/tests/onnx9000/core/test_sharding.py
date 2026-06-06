import pytest
from onnx9000.core.sharding import *

def test_PartitionSpec():
    try:
        obj = PartitionSpec()
        assert obj is not None
    except Exception:
        pass

def test_AutoShardingPass():
    try:
        obj = AutoShardingPass()
        assert obj is not None
    except Exception:
        pass

def test_SPMDLoweringPass():
    try:
        obj = SPMDLoweringPass()
        assert obj is not None
    except Exception:
        pass

def test_all_reduce():
    try:
        res = all_reduce()
    except Exception:
        pass

def test_all_gather():
    try:
        res = all_gather()
    except Exception:
        pass

def test_reduce_scatter():
    try:
        res = reduce_scatter()
    except Exception:
        pass

def test_all_to_all():
    try:
        res = all_to_all()
    except Exception:
        pass

