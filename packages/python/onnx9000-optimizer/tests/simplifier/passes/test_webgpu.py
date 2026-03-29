"""Tests for packages/python/onnx9000-optimizer/tests/simplifier/passes/test_webgpu.py."""

import numpy as np
import pytest


def test_polyfill_webgpu():
    """Test polyfill webgpu."""
    from onnx9000.optimizer.simplifier.passes.webgpu import polyfill_webgpu_unsupported

    class MockNode:
        """MockNode implementation."""

        def __init__(self, op_type):
            """Perform   init   operation."""
            self.op_type = op_type

    class MockGraph:
        """MockGraph implementation."""

        def __init__(self):
            """Perform   init   operation."""
            self.nodes = [MockNode("UnsupportedOpX"), MockNode("Conv")]

    g = MockGraph()
    changed = polyfill_webgpu_unsupported(g)
    assert changed


def test_optimize_for_webgpu():
    """Test optimize for webgpu."""
    from onnx9000.optimizer.simplifier.passes.webgpu import optimize_for_webgpu

    class MockNode:
        """MockNode implementation."""

        def __init__(self, op_type):
            """Perform   init   operation."""
            self.op_type = op_type

    class MockGraph:
        """MockGraph implementation."""

        def __init__(self):
            """Perform   init   operation."""
            self.nodes = [MockNode("UnsupportedOpX"), MockNode("Conv")]

    g = MockGraph()
    changed = optimize_for_webgpu(g)
    assert changed


def test_fp16_cast():
    """Test fp16 cast."""
    from onnx9000.optimizer.simplifier.passes.webgpu import fp16_cast

    class MockInit:
        """MockInit implementation."""

        def __init__(self, name, dtype):
            """Perform   init   operation."""
            self.name = name
            self.dtype = dtype
            self.data = b"abc"

        def numpy(self):
            """Perform numpy operation."""
            return np.array([1.0, 2.0], dtype=np.float32)

    class MockNode:
        """MockNode implementation."""

        def __init__(self, op_type, inputs):
            """Perform   init   operation."""
            self.op_type = op_type
            self.inputs = inputs

    class MockGraph:
        """MockGraph implementation."""

        def __init__(self):
            """Perform   init   operation."""
            self.nodes = [MockNode("LayerNormalization", ["init1"]), MockNode("Conv", ["init2"])]
            self.initializers = {
                "init1": MockInit("init1", "float32"),
                "init2": MockInit("init2", "float32"),
            }

    g = MockGraph()
    changed = fp16_cast(g)
    assert changed
    assert g.initializers["init2"].dtype == "float16"


def test_generate_reports(tmp_path):
    """Test generate reports."""
    from onnx9000.optimizer.simplifier.passes.webgpu import (
        generate_execution_schedule,
        generate_html_report,
    )

    class MockNode:
        """MockNode implementation."""

        def __init__(self, name, op_type, inputs, outputs):
            """Perform   init   operation."""
            self.name = name
            self.op_type = op_type
            self.inputs = inputs
            self.outputs = outputs

    class MockGraph:
        """MockGraph implementation."""

        def __init__(self):
            """Perform   init   operation."""
            self.nodes = [MockNode("A", "Conv", ["a"], ["b"])]

    g = MockGraph()
    out_html = tmp_path / "report.html"
    generate_html_report(g, g, str(out_html))
    assert out_html.exists()
    out_json = tmp_path / "schedule.json"
    generate_execution_schedule(g, str(out_json))
    assert out_json.exists()


def test_fuse_methods():
    """Test fuse methods."""
    from onnx9000.optimizer.simplifier.passes.webgpu import (
        fuse_geglu,
        fuse_swiglu,
        inject_web_worker_boundaries,
        replace_gather_with_lookup,
    )

    assert fuse_swiglu(None) is False
    assert fuse_geglu(None) is False
    assert replace_gather_with_lookup(None) is False
    assert inject_web_worker_boundaries(None) is False
