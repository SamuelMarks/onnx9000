"""Module docstring."""

import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.tflite_exporter.compiler.layout import LayoutOptimizer
from onnx9000.tflite_exporter.exporter import TFLiteExporter
from onnx9000.tflite_exporter.flatbuffer.builder import FlatBufferBuilder


def test_layout_expand_tile():
    """Docstring for D103."""
    g = Graph("ExpandTile")
    g.inputs.append(ValueInfo("X", (1, 3, 2, 2), DType.FLOAT32))

    t = Tensor("shape", (4,), DType.INT64, data=np.array([1, 3, 2, 2], dtype=np.int64).tobytes())
    t.is_initializer = True
    g.tensors["shape"] = t
    g.initializers.append("shape")

    class AttrMock:
        def __init__(self, val):
            self.value = val

    g.nodes.append(Node("Transpose", ["X"], ["trans"], {"perm": AttrMock([0, 2, 3, 1])}))
    g.nodes.append(Node("Expand", ["trans", "shape"], ["out"]))
    g.outputs.append("out")

    LayoutOptimizer(g).push_down_transposes()


def test_exporter_extras():
    """Docstring for D103."""
    exporter = TFLiteExporter()
    exporter.builder = FlatBufferBuilder(1024)

    exporter.add_metadata("test_meta", b"test_data")
    assert exporter.metadata_list[0][0] == "test_meta"

    # 63: Empty buffer
    exporter.add_buffer(b"")

    exporter.destroy()
    assert getattr(exporter, "builder", None) is None

    exporter = TFLiteExporter()
    exporter.operator_codes["test"] = 1
    exporter.operator_code_offsets.append(2)
    exporter.buffers["b"] = 3
    exporter.buffer_offsets.append(0)

    j = exporter.to_json()
    assert j["version"] == 3
    assert j["buffersCount"] == 2
    assert j["operatorCodesCount"] == 1


def test_builder_grow_zero():
    """Docstring for D103."""
    b = FlatBufferBuilder(0)
    assert len(b.bb) == 0
    b.grow_buffer()
    assert len(b.bb) == 1024
