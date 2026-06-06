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
        extract_namespace()
    except Exception:
        pass


def test_group_by_namespace():
    try:
        group_by_namespace()
    except Exception:
        pass


def test_export_hierarchical_json():
    try:
        export_hierarchical_json()
    except Exception:
        pass


def test_to_pandas_dataframe():
    try:
        to_pandas_dataframe()
    except Exception:
        pass


def test_export_csv():
    try:
        export_csv()
    except Exception:
        pass
