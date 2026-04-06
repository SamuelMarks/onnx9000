"""Tests for Keras 3 parser coverage gaps."""

import sys
from unittest.mock import MagicMock

import pytest

# Mock keras and tensorflow
keras_mock = MagicMock()
sys.modules["keras"] = keras_mock
sys.modules["tensorflow"] = MagicMock()

from onnx9000.converters.tf.keras_v3_parser import Keras3Parser


def create_mock_layer(name="layer", num_inbound=1):
    layer = MagicMock()
    layer.name = name
    layer.get_config.return_value = {"units": 5}

    # default module
    layer.__class__.__module__ = "keras.layers"
    layer.__class__.__name__ = "Dense"

    w = MagicMock()
    w.name = "kernel"
    w.numpy.return_value = 1.0
    w.dtype = "float32"
    layer.weights = [w]
    layer._inbound_nodes = []

    for _ in range(num_inbound):
        node = MagicMock()
        in_tensor = MagicMock()
        in_tensor.name = "in_tensor"
        out_tensor = MagicMock()
        out_tensor.name = "out_tensor"
        node.input_tensors = [in_tensor]
        node.output_tensors = [out_tensor]
        layer._inbound_nodes.append(node)

    return layer


def test_keras3_parser_basic():
    """Test basic Keras 3 functional model parsing."""
    model = MagicMock()
    model.built = True
    model.operations = [create_mock_layer("Dense")]

    parser = Keras3Parser(model)
    graph = parser.parse()
    assert len(graph.nodes) > 0
    assert any(n.name == "Dense" for n in graph.nodes)


def test_keras3_parser_input_layer():
    """Test InputLayer conversion."""
    model = MagicMock()
    model.built = True
    inp = create_mock_layer("Input")
    inp.__class__.__name__ = "InputLayer"
    model.operations = [inp]
    parser = Keras3Parser(model)
    graph = parser.parse()
    assert any(n.op == "Placeholder" for n in graph.nodes)


def test_keras3_parser_subclassed_tracing():
    """Test tracing a subclassed model."""
    model = MagicMock()
    model.built = False

    def model_call(inputs):
        return inputs

    model.__call__ = model_call

    # We mock keras.Input and keras.Model to return a new mock model that IS built
    built_model = MagicMock()
    built_model.built = True
    built_model.operations = [create_mock_layer("TracedDense")]

    keras_mock.Input = MagicMock(return_value=MagicMock())
    keras_mock.Model = MagicMock(return_value=built_model)

    parser = Keras3Parser(model, input_shape=(None, 10))
    graph = parser.parse()
    assert len(graph.nodes) > 0


def test_keras3_parser_subclassed_multi_input():
    """Test tracing a subclassed model with multiple inputs."""
    model = MagicMock()
    model.built = False

    def model_call(inputs):
        return inputs

    model.__call__ = model_call

    built_model = MagicMock()
    built_model.built = True
    built_model.operations = [create_mock_layer("MultiInput")]

    keras_mock.Input = MagicMock(return_value=MagicMock())
    keras_mock.Model = MagicMock(return_value=built_model)

    parser = Keras3Parser(model, input_shape=[(None, 10), (None, 10)])
    graph = parser.parse()
    assert len(graph.nodes) > 0


def test_keras3_parser_no_input_shape():
    """Test error when no input shape is provided for subclassed model."""
    model = MagicMock()
    model.built = False
    model.input_shape = None

    parser = Keras3Parser(model)
    with pytest.raises(
        ValueError, match="Subclassed Keras 3 model must be built or have input_shape to be parsed."
    ):
        parser.parse()


def test_keras3_parser_reused_layer():
    """Test parsing a model with reused layers (multiple inbound nodes)."""
    model = MagicMock()
    model.built = True
    model.operations = [create_mock_layer("Shared", num_inbound=2)]

    parser = Keras3Parser(model)
    graph = parser.parse()
    # shared layer should have 2 nodes
    shared_nodes = [n for n in graph.nodes if "Shared" in n.name]
    assert len(shared_nodes) >= 2


def test_keras3_parser_get_tensor_name():
    """Test get_tensor_name with various tensor types."""
    parser = Keras3Parser(None)

    class MockTensor:
        """Mock tensor."""

        def __init__(self, name):
            """Init."""
            self.name = name

    t1 = MockTensor("real_name")
    assert parser.get_tensor_name(t1) == "real_name"

    t2 = MockTensor("keras_tensor_0")
    name2 = parser.get_tensor_name(t2)
    assert name2.startswith("tensor_")

    # Cached
    assert parser.get_tensor_name(t2) == name2


def test_keras3_parser_module_path_else():
    """Test op_type generation for non-keras module."""
    model = MagicMock()
    model.built = True

    custom_layer = create_mock_layer("Custom")
    custom_layer.__class__.__module__ = "my.custom.module"
    custom_layer.__class__.__name__ = "MyLayer"

    model.operations = [custom_layer]

    parser = Keras3Parser(model)
    graph = parser.parse()
    assert len(graph.nodes) > 0
    assert graph.nodes[1].op == "my.custom.module.MyLayer"
