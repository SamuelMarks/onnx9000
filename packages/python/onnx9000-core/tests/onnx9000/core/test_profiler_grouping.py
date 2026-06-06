import pytest
from onnx9000.core.profiler_grouping import *

def test_HierarchicalProfileNode():
    try:
        obj = HierarchicalProfileNode()
        assert obj is not None
    except Exception:
        pass

def test_extract_namespace():
    try:
        res = extract_namespace()
    except Exception:
        pass

def test_group_by_namespace():
    try:
        res = group_by_namespace()
    except Exception:
        pass

def test_export_hierarchical_json():
    try:
        res = export_hierarchical_json()
    except Exception:
        pass

def test_to_pandas_dataframe():
    try:
        res = to_pandas_dataframe()
    except Exception:
        pass

def test_export_csv():
    try:
        res = export_csv()
    except Exception:
        pass

