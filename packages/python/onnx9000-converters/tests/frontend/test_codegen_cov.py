"""Tests for frontend codegen."""

from onnx9000.converters.frontend.builder import GraphBuilder
from onnx9000.converters.frontend.codegen import generate_pytorch, generate_keras, generate_jax
from onnx9000.converters.frontend.tensor import Node, Tensor
from onnx9000.core.dtypes import DType


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
    assert "class MyModel(nn.Module):" in code
    assert "nn.Parameter" in code

    builder.nodes.append(Node(op_type="Mul", inputs=[in1, p1], outputs=[out]))
    builder.nodes.append(Node(op_type="Sub", inputs=[in1, p1], outputs=[out]))
    builder.nodes.append(Node(op_type="Div", inputs=[in1, p1], outputs=[out]))
    builder.nodes.append(Node(op_type="MatMul", inputs=[in1, p1], outputs=[out]))
    builder.nodes.append(Node(op_type="Relu", inputs=[in1], outputs=[out]))
    builder.nodes.append(Node(op_type="Abs", inputs=[in1], outputs=[out]))
    builder.doc_string = None
    code = generate_pytorch(builder)
    assert "Generated PyTorch module" in code


def test_codegen_pytorch_no_outputs():
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
    assert "def MyKerasModel_model():" in code
    assert "import keras" in code


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
    assert "def MyJAXModel_func(params, inputs):" in code
    assert "jax.numpy" in code


def test_codegen_jax_no_outputs():
    builder = GraphBuilder(name="MyModel")
    assert "return None" in generate_jax(builder)

    out1 = Tensor(name="o1", shape=(1,), dtype=DType.FLOAT32)
    out2 = Tensor(name="o2", shape=(1,), dtype=DType.FLOAT32)
    builder.outputs.extend([out1, out2])
    assert "return (o1, o2)" in generate_jax(builder)
