"""Code generation for onnx9000."""

from onnx9000.converters.frontend.builder import GraphBuilder
from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
from onnx9000.core.codegen.keras import ONNXToKerasVisitor


def generate_pytorch(builder: GraphBuilder) -> str:
    """Generate PyTorch nn.Module source code from a GraphBuilder."""
    return ONNXToPyTorchVisitor(builder.to_graph()).generate()


def generate_keras(builder: GraphBuilder) -> str:
    """Generate Keras source code from a GraphBuilder."""
    return ONNXToKerasVisitor(builder.to_graph()).generate()


def generate_jax(builder: GraphBuilder) -> str:
    """Generate JAX source code from a GraphBuilder."""
    return ONNXToFlaxNNXVisitor(builder.to_graph()).generate()
