"""Comprehensive unit tests for onnx9000.pytorch_codegen and onnx9000_pytorch_codegen."""

import onnx9000_pytorch_codegen
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.pytorch_codegen import compile_to_pytorch, run


def test_pytorch_codegen_compile_and_run() -> None:
    """Verify run() and compile_to_pytorch() generating valid PyTorch module code."""
    assert run() == "[pytorch-codegen] processed"
    assert onnx9000_pytorch_codegen.run() == "[pytorch-codegen] processed"

    g = Graph("TorchNet")
    x = Tensor("X", (1, 4), DType.FLOAT32)
    y = Tensor("Y", (1, 4), DType.FLOAT32)
    g.add_tensor(x)
    g.add_tensor(y)
    g.inputs.append(ValueInfo("X", (1, 4), DType.FLOAT32))
    g.outputs.append(ValueInfo("Y", (1, 4), DType.FLOAT32))
    g.add_node(Node("Relu", [x], [y]))

    code = compile_to_pytorch(g)
    assert "import torch" in code
    assert "class Model_TorchNet(nn.Module):" in code
