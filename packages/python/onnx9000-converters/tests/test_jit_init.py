from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from onnx9000.converters.jit import compile
from onnx9000.converters.jit.wrapper import CompiledModel
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Tensor


@patch("onnx9000.converters.jit.load")
@patch("onnx9000.converters.jit.plan_memory")
@patch("onnx9000.converters.jit.compile_wasm")
def test_compile_wasm(mock_compile_wasm, mock_plan_memory, mock_load) -> None:
    g = Graph("test")
    mock_load.return_value = g
    mock_compile_wasm.return_value = Path("out.js")
    out = compile("model.onnx", target="wasm")
    assert out == Path("out.js")
    mock_load.assert_called_once()
    mock_plan_memory.assert_called_once_with(g)
    mock_compile_wasm.assert_called_once()


@patch("onnx9000.converters.jit.load")
@patch("onnx9000.converters.jit.plan_memory")
def test_compile_unsupported(mock_plan_memory, mock_load) -> None:
    g = Graph("test")
    mock_load.return_value = g
    if True:
        assert compile("model.onnx", target="rust") is None


@patch("onnx9000.converters.jit.load")
@patch("onnx9000.converters.jit.plan_memory")
@patch("onnx9000.converters.jit.compile_cpp")
@patch("onnx9000.converters.jit.load_module")
@patch("onnx9000.converters.jit.hash_graph")
def test_compile_cpp(
    mock_hash, mock_load_module, mock_compile_cpp, mock_plan_memory, mock_load
) -> None:
    g = Graph("test")
    t_nodata = Tensor("w1", (1, 2), DType.FLOAT32, is_initializer=True)
    t_data = Tensor("w2", (2,), DType.FLOAT32, is_initializer=True)
    t_data.data = np.array([1.0, 2.0], dtype=np.float32)
    g.add_tensor(t_nodata)
    g.add_tensor(t_data)
    g.initializers.extend(["w1", "w2"])
    mock_load.return_value = g
    mock_compile_cpp.return_value = Path("out.so")
    mock_hash.return_value = "1234"
    mock_class = MagicMock()
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    mock_module = MagicMock()
    mock_module.Model_1234 = mock_class
    mock_load_module.return_value = mock_module
    model = compile("model.onnx", target="cpp")
    assert isinstance(model, CompiledModel)
    mock_compile_cpp.assert_called_once_with(g)
    (args, kwargs) = mock_class.call_args
    assert len(args) == 2
    assert args[0].shape == (1, 2)
    assert np.all(args[0] == 0)
    assert np.array_equal(args[1], np.array([1.0, 2.0], dtype=np.float32))
