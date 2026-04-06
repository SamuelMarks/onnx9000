"""Test code generation logic."""

from onnx9000.converters.frontend.builder import GraphBuilder
from onnx9000.converters.frontend.codegen import generate_jax, generate_keras, generate_pytorch
from onnx9000.converters.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Node


def test_codegen_pytorch():
    """Test PyTorch codegen."""
    builder = GraphBuilder(name="MyModel")
    builder.doc_string = "My doc"
    in1 = Tensor(name="x", shape=(1,), dtype=DType.FLOAT32)
    p1 = Tensor(name="w", shape=(1,), dtype=DType.FLOAT32, is_buffer=False)
    p2 = Tensor(name="b", shape=(1,), dtype=DType.FLOAT32, is_buffer=True)
    out = Tensor(name="y", shape=(1,), dtype=DType.FLOAT32)

    builder.inputs.append(in1)
    builder.parameters.extend([p1, p2])
    builder.outputs.append(out)

    node = Node(op_type="Add", inputs=[in1, p1], outputs=[out])
    builder.nodes.append(node)

    code = generate_pytorch(builder)
    assert "class Model_MyModel(nn.Module):" in code
    assert "def forward(self, x):" in code


def test_codegen_pytorch_no_outputs():
    """Docstring for D103."""
    builder = GraphBuilder(name="MyModel")
    assert "return None" in generate_pytorch(builder)

    out1 = Tensor(name="o1", shape=(1,), dtype=DType.FLOAT32)
    out2 = Tensor(name="o2", shape=(1,), dtype=DType.FLOAT32)
    builder.outputs.extend([out1, out2])
    assert "return (o1, o2)" in generate_pytorch(builder)


def test_codegen_keras():
    """Test Keras codegen."""
    builder = GraphBuilder(name="MyKerasModel")
    in1 = Tensor(name="x", shape=(1,), dtype=DType.FLOAT32)
    builder.inputs.append(in1)

    code = generate_keras(builder)
    assert "class Model_MyKerasModel(keras.Model):" in code


def test_codegen_jax():
    """Test JAX codegen."""
    builder = GraphBuilder(name="MyJAXModel")
    in1 = Tensor(name="x", shape=(1,), dtype=DType.FLOAT32)
    p1 = Tensor(name="w", shape=(1,), dtype=DType.FLOAT32, is_buffer=False)
    out = Tensor(name="y", shape=(1,), dtype=DType.FLOAT32)

    builder.inputs.append(in1)
    builder.parameters.append(p1)
    builder.outputs.append(out)

    builder.nodes.append(Node(op_type="Add", inputs=[in1, p1], outputs=[out]))
    builder.nodes.append(Node(op_type="Mul", inputs=[in1, p1], outputs=[out]))
    builder.nodes.append(Node(op_type="Relu", inputs=[in1], outputs=[out]))
    builder.nodes.append(Node(op_type="MatMul", inputs=[in1, p1], outputs=[out]))
    builder.nodes.append(Node(op_type="Abs", inputs=[in1], outputs=[out]))

    code = generate_jax(builder)
    assert "class Model_MyJAXModel(nnx.Module):" in code


def test_codegen_jax_no_outputs():
    """Docstring for D103."""
    builder = GraphBuilder(name="MyModel")
    assert "return None" in generate_jax(builder)

    out1 = Tensor(name="o1", shape=(1,), dtype=DType.FLOAT32)
    out2 = Tensor(name="o2", shape=(1,), dtype=DType.FLOAT32)
    builder.outputs.extend([out1, out2])
    assert "return (o1, o2)" in generate_jax(builder)
