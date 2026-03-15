"""Module providing core logic and structural definitions."""

import pytest
import logging
from pathlib import Path
import numpy as np
from onnx9000.core.ir import DynamicDim, Tensor, Node, Graph
from onnx9000.core.dtypes import DType, to_cpp_type, to_emscripten_type
from onnx9000.core.exceptions import (
    Onnx9000Error,
    CompilationError,
    UnsupportedOpError,
    ShapeMismatchError,
)
from onnx9000.core import config
from onnx9000.core.registry import OperatorRegistry, registry
from onnx9000.utils.logger import get_logger
from onnx9000.utils import cache


def test_dynamic_dim():
    """Provides semantic functionality and verification."""
    d1 = DynamicDim("batch")
    d2 = DynamicDim("batch")
    d3 = DynamicDim(-1)
    assert repr(d1) == "DynamicDim(batch)"
    assert str(d1) == "batch"
    assert d1 == d2
    assert d1 != d3
    assert d1 != "batch"


def test_tensor():
    """Provides semantic functionality and verification."""
    t = Tensor(
        "my_tensor",
        (1, DynamicDim("seq")),
        DType.FLOAT32,
        is_initializer=True,
        requires_grad=False,
        data=np.array([1.0]),
    )
    r = repr(t)
    assert "ir.Tensor" in r
    assert "buf=None" in r
    assert t.lifespan == (-1, -1)


def test_node():
    """Provides semantic functionality and verification."""
    n = Node("Relu", ["in"], ["out"], {"alpha": 0.01}, name="relu1")
    r = repr(n)
    assert "ir.Node(Relu, ['in'] -> ['out'])" in r


def test_graph(caplog):
    """Provides semantic functionality and verification."""
    g = Graph("my_graph")
    t = Tensor("t1", (1,), DType.FLOAT32)
    n = Node("Add", ["a", "b"], ["t1"], {})
    g.add_tensor(t)
    g.add_node(n)
    g.inputs = ["a", "b"]
    g.outputs = ["t1"]
    with caplog.at_level(logging.INFO):
        g.print_visualizer()
    assert "=== Graph: my_graph ===" in caplog.text
    assert "Inputs: ['a', 'b']" in caplog.text
    assert "Outputs: ['t1']" in caplog.text


def test_dtypes():
    """Provides semantic functionality and verification."""
    assert to_cpp_type(DType.FLOAT32) == "float"
    assert to_cpp_type(DType.INT64) == "int64_t"
    with pytest.raises(ValueError, match="No C\\+\\+ type mapped"):
        to_cpp_type(DType.UNDEFINED)
    assert to_emscripten_type(DType.FLOAT32) == "Float32Array"
    assert to_emscripten_type(DType.INT64) == "BigInt64Array"
    with pytest.raises(ValueError, match="No Emscripten TypedArray mapped"):
        to_emscripten_type(DType.UNDEFINED)


def test_exceptions():
    """Provides semantic functionality and verification."""
    e = UnsupportedOpError("MagicOp")
    assert e.op_type == "MagicOp"
    assert "Operator 'MagicOp' is not supported yet." in str(e)
    e2 = UnsupportedOpError("MagicOp", "Custom msg")
    assert "Custom msg" in str(e2)


def test_config():
    """Provides semantic functionality and verification."""
    assert hasattr(config, "ONNX9000_CACHE_DIR")
    assert hasattr(config, "ONNX9000_COMPILER")
    assert hasattr(config, "ONNX9000_WASM_COMPILER")


def test_registry():
    """Provides semantic functionality and verification."""
    reg = OperatorRegistry()

    @reg.register("Add")
    def gen_add():
        """Provides gen add functionality and verification."""
        return "add"

    assert reg.get_generator("Add")() == "add"
    with pytest.raises(ValueError, match="is already registered"):

        @reg.register("Add")
        def gen_add2():
            """Provides gen add2 functionality and verification."""
            return "add2"

    with pytest.raises(UnsupportedOpError):
        reg.get_generator("Sub")
    with pytest.raises(ModuleNotFoundError):
        reg.load_plugin("some_non_existent_module")


def test_logger():
    """Provides semantic functionality and verification."""
    l1 = get_logger("test_logger")
    l2 = get_logger("test_logger")
    assert l1 is l2


def test_cache(monkeypatch, tmp_path):
    """Provides semantic functionality and verification."""
    monkeypatch.setattr(config, "ONNX9000_CACHE_DIR", tmp_path / "cache")
    cache.clear_cache()
    c_dir = config.ONNX9000_CACHE_DIR
    c_dir.mkdir(parents=True)
    cache.clear_cache()
    assert not c_dir.exists()
    c_dir.mkdir()

    def mock_rmtree(path):
        """Provides mock rmtree functionality and verification."""
        raise OSError("Permission denied")

    monkeypatch.setattr(cache.shutil, "rmtree", mock_rmtree)
    cache.clear_cache()
