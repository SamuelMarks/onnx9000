"""Tests the profiler grouping more module functionality."""

import os

import pytest
from onnx9000.core.profiler_grouping import (
    HierarchicalProfileNode,
    export_csv,
    extract_namespace,
)


class DummyProfilerResult:
    """Represents the Dummy Profiler Result class."""

    def __init__(self, node_profiles):
        """Initialize the instance."""
        self.node_profiles = node_profiles


def test_extract_namespace():
    """Tests the extract namespace functionality."""
    assert extract_namespace("a/b/c") == ["a", "b", "c"]
    assert extract_namespace("flat") == ["flat"]


def test_print_tree(capsys):
    """Tests the print tree functionality."""
    node = HierarchicalProfileNode("root")
    node.flops = 10
    child = HierarchicalProfileNode("child")
    child.flops = 5
    node.children["child"] = child
    node.print_tree()
    out, err = capsys.readouterr()
    assert "- root: FLOPs=10" in out
    assert "- child: FLOPs=5" in out


def test_export_csv_empty(tmp_path):
    """Tests the export csv empty functionality."""
    f_path = str(tmp_path / "out.csv")
    res = DummyProfilerResult([])
    export_csv(res, f_path)
    assert not os.path.exists(f_path)
