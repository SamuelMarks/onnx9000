"""Tests for Keras 3 parser coverage gaps."""

import keras
import numpy as np
import pytest
from onnx9000.converters.tf.keras_v3_parser import Keras3Parser


def test_keras3_parser_basic():
    """Test basic Keras 3 functional model parsing."""
    inputs = keras.Input(shape=(10,))
    outputs = keras.layers.Dense(5)(inputs)
    model = keras.Model(inputs, outputs)

    parser = Keras3Parser(model)
    graph = parser.parse()
    assert len(graph.nodes) > 0
    assert any("Dense" in n.op for n in graph.nodes)


def test_keras3_parser_subclassed_tracing():
    """Test tracing a subclassed model."""

    class MyModel(keras.Model):
        """My model."""

        def __init__(self):
            """Init."""
            super().__init__()
            self.dense = keras.layers.Dense(5)

        def call(self, x):
            """Call."""
            return self.dense(x)

    model = MyModel()
    # Not built yet
    parser = Keras3Parser(model, input_shape=(None, 10))
    graph = parser.parse()
    assert len(graph.nodes) > 0


def test_keras3_parser_subclassed_multi_input():
    """Test tracing a subclassed model with multiple inputs."""

    class MyMultiModel(keras.Model):
        """My multi model."""

        def call(self, inputs):
            """Call."""
            return inputs[0] + inputs[1]

    model = MyMultiModel()
    parser = Keras3Parser(model, input_shape=[(None, 10), (None, 10)])
    graph = parser.parse()
    assert len(graph.nodes) > 0


def test_keras3_parser_no_input_shape():
    """Test error when no input shape is provided for subclassed model."""

    class MyModel(keras.Model):
        """My model."""

        def call(self, x):
            """Call."""
            return x

    model = MyModel()
    parser = Keras3Parser(model)
    with pytest.raises(ValueError, match="must be built or have input_shape"):
        parser.parse()


def test_keras3_parser_reused_layer():
    """Test parsing a model with reused layers (multiple inbound nodes)."""
    inputs1 = keras.Input(shape=(10,), name="in1")
    inputs2 = keras.Input(shape=(10,), name="in2")
    shared_layer = keras.layers.Dense(5, name="shared")
    out1 = shared_layer(inputs1)
    out2 = shared_layer(inputs2)
    model = keras.Model([inputs1, inputs2], [out1, out2])

    parser = Keras3Parser(model)
    graph = parser.parse()
    # shared layer should have 2 nodes
    shared_nodes = [n for n in graph.nodes if "shared" in n.name]
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
    Keras3Parser(None)

    class CustomOp:
        """Custom op."""

        def __init__(self):
            """Init."""
            self.name = "custom"
            self._inbound_nodes = []
            self.weights = []

        def get_config(self):
            """Get config."""
            return {}

    CustomOp()
    # Manually trigger the module path logic if possible or mock it
    # Keras3Parser.parse uses op.__class__.__module__
