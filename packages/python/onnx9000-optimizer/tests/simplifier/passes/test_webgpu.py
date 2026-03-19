import pytest
import numpy as np


def test_polyfill_webgpu():
    from onnx9000.optimizer.simplifier.passes.webgpu import polyfill_webgpu_unsupported

    class MockNode:
        def __init__(self, op_type):
            self.op_type = op_type

    class MockGraph:
        def __init__(self):
            self.nodes = [MockNode("UnsupportedOpX"), MockNode("Conv")]

    g = MockGraph()
    changed = polyfill_webgpu_unsupported(g)
    assert changed


def test_optimize_for_webgpu():
    from onnx9000.optimizer.simplifier.passes.webgpu import optimize_for_webgpu

    class MockNode:
        def __init__(self, op_type):
            self.op_type = op_type

    class MockGraph:
        def __init__(self):
            self.nodes = [MockNode("UnsupportedOpX"), MockNode("Conv")]

    g = MockGraph()
    changed = optimize_for_webgpu(g)
    assert changed


def test_fp16_cast():
    from onnx9000.optimizer.simplifier.passes.webgpu import fp16_cast

    class MockInit:
        def __init__(self, name, dtype):
            self.name = name
            self.dtype = dtype
            self.data = b"abc"

        def numpy(self):
            return np.array([1.0, 2.0], dtype=np.float32)

    class MockNode:
        def __init__(self, op_type, inputs):
            self.op_type = op_type
            self.inputs = inputs

    class MockGraph:
        def __init__(self):
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
    from onnx9000.optimizer.simplifier.passes.webgpu import (
        generate_html_report,
        generate_execution_schedule,
    )

    class MockNode:
        def __init__(self, name, op_type, inputs, outputs):
            self.name = name
            self.op_type = op_type
            self.inputs = inputs
            self.outputs = outputs

    class MockGraph:
        def __init__(self):
            self.nodes = [MockNode("A", "Conv", ["a"], ["b"])]

    g = MockGraph()
    out_html = tmp_path / "report.html"
    generate_html_report(g, g, str(out_html))
    assert out_html.exists()

    out_json = tmp_path / "schedule.json"
    generate_execution_schedule(g, str(out_json))
    assert out_json.exists()


def test_fuse_methods():
    from onnx9000.optimizer.simplifier.passes.webgpu import (
        fuse_swiglu,
        fuse_geglu,
        replace_gather_with_lookup,
        inject_web_worker_boundaries,
    )

    assert fuse_swiglu(None) is False
    assert fuse_geglu(None) is False
    assert replace_gather_with_lookup(None) is False
    assert inject_web_worker_boundaries(None) is False
