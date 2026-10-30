"""Tests for checker."""

import asyncio

import pytest
from onnx9000.core.checker import (
    SchemaRegistry,
    ValidationContext,
    _check_op_specific,
    check_attribute,
    check_model,
    check_model_async,
    check_tensor,
)
from onnx9000.core.exceptions import (
    UnsupportedOpError,
    UnsupportedOpsetError,
    ValidationError,
)


class MockTensor:
    """Represents mock tensor."""

    def __init__(
        self,
        data_type: int | float | str | bool | bytes | list | dict | tuple | None,
        shape: int | float | str | bool | bytes | list | dict | tuple | None,
        is_initializer: int | float | str | bool | bytes | list | dict | tuple | None = False,
        raw_data: int | float | str | bool | bytes | list | dict | tuple | None = None,
        data_location: int | float | str | bool | bytes | list | dict | tuple | None = None,
        external_data: int | float | str | bool | bytes | list | dict | tuple | None = None,
    ) -> None:
        """Initialize MockTensor with configured parameters and state.

        Args:
            data_type: Parameter specifying data type.
            shape: Parameter specifying shape.
            is_initializer: Parameter specifying is initializer.
            raw_data: Parameter specifying raw data.
            data_location: Parameter specifying data location.
            external_data: Parameter specifying external data.
        """
        self.data_type = data_type
        self.shape = shape
        self.is_initializer = is_initializer
        self.raw_data = raw_data
        self.data_location = data_location
        self.external_data = external_data


def test_validation_context() -> None:
    """Verify that validation context behaves as expected."""
    ctx = ValidationContext()
    assert ctx.strict is True
    assert ctx.errors == []


def test_schema_registry() -> None:
    """Verify that schema registry behaves as expected."""
    registry = SchemaRegistry()
    assert registry.get_schema("Conv", 21) == {"pads": "ints", "strides": "ints"}

    with pytest.raises(UnsupportedOpsetError):
        registry.get_schema("Conv", 99)

    with pytest.raises(UnsupportedOpError):
        registry.get_schema("UnknownOp", 21)

    registry.register_custom_schema("custom", 1, {"MyOp": {}})
    assert registry.get_schema("MyOp", 1, "custom") == {}


def test_check_tensor() -> None:
    """Verify that check tensor behaves as expected."""
    ctx = ValidationContext()

    # Valid tensor
    t = MockTensor("float", [1, 2, 3])
    check_tensor(t, ctx)
    assert not ctx.errors

    # Invalid dtype
    t2 = MockTensor("invalid", [1])
    check_tensor(t2, ctx)
    assert "Invalid data_type: invalid" in ctx.errors
    ctx.errors.clear()

    # Invalid dim
    t3 = MockTensor("float", [-2])
    check_tensor(t3, ctx)
    assert "Invalid dim: -2" in ctx.errors
    ctx.errors.clear()

    # Initializer with -1
    t4 = MockTensor("float", [-1], is_initializer=True)
    check_tensor(t4, ctx)
    assert "Initializer cannot have -1 dim" in ctx.errors
    ctx.errors.clear()

    # External data
    t5 = MockTensor("float", [1], data_location="EXTERNAL")
    check_tensor(t5, ctx)
    assert "External data missing" in ctx.errors
    ctx.errors.clear()

    t6 = MockTensor("float", [1], data_location="EXTERNAL", external_data={"location": "../test"})
    check_tensor(t6, ctx)
    assert "Directory traversal not allowed in external data" in ctx.errors
    ctx.errors.clear()

    # Raw data > 2GB (mock it)
    t7 = MockTensor("float", [1024, 1024, 1024], raw_data=b"0" * (2 * 1024 * 1024 * 1024 + 1))
    check_tensor(t7, ctx)
    assert "Tensor exceeds 2GB" in ctx.errors
    ctx.errors.clear()


def test_check_attribute() -> None:
    """Verify that check attribute behaves as expected."""
    ctx = ValidationContext()
    check_attribute("pads", [1, 1], "ints", ctx)
    assert not ctx.errors

    check_attribute("pads", ["1"], "ints", ctx)
    assert "Expected ints for pads" in ctx.errors
    ctx.errors.clear()

    check_attribute("scales", [1.0, 2.0], "floats", ctx)
    assert not ctx.errors

    check_attribute("scales", ["1.0"], "floats", ctx)
    assert "Expected floats for scales" in ctx.errors
    ctx.errors.clear()


class MockNode:
    """Represents mock node."""

    def __init__(
        self,
        op_type: int | float | str | bool | bytes | list | dict | tuple | None,
        inputs: int | float | str | bool | bytes | list | dict | tuple | None,
        outputs: int | float | str | bool | bytes | list | dict | tuple | None,
        attributes: int | float | str | bool | bytes | list | dict | tuple | None = None,
    ) -> None:
        """Initialize MockNode with configured parameters and state.

        Args:
            op_type: Parameter specifying op type.
            inputs: Parameter specifying inputs.
            outputs: Parameter specifying outputs.
            attributes: Parameter specifying attributes.
        """
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes or {}


class MockModel:
    """Represents mock model."""

    def __init__(
        self,
        ir_version: int | float | str | bool | bytes | list | dict | tuple | None = 8,
        producer_name: int | float | str | bool | bytes | list | dict | tuple | None = "test",
        opset_import: int | float | str | bool | bytes | list | dict | tuple | None = None,
        graph: int | float | str | bool | bytes | list | dict | tuple | None = None,
    ) -> None:
        """Initialize MockModel with configured parameters and state.

        Args:
            ir_version: Parameter specifying ir version.
            producer_name: Parameter specifying producer name.
            opset_import: Parameter specifying opset import.
            graph: Parameter specifying graph.
        """
        self.ir_version = ir_version
        self.producer_name = producer_name
        self.opset_import = opset_import or []
        self.graph = graph


class MockDomain:
    """Represents mock domain."""

    def __init__(
        self, domain: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> None:
        """Initialize MockDomain with configured parameters and state.

        Args:
            domain: Parameter specifying domain.
        """
        self.domain = domain


class MockGraph:
    """Represents mock graph."""

    def __init__(
        self,
        inputs: int | float | str | bool | bytes | list | dict | tuple | None = None,
        initializers: int | float | str | bool | bytes | list | dict | tuple | None = None,
        nodes: int | float | str | bool | bytes | list | dict | tuple | None = None,
    ) -> None:
        """Initialize MockGraph with configured parameters and state.

        Args:
            inputs: Parameter specifying inputs.
            initializers: Parameter specifying initializers.
            nodes: Parameter specifying nodes.
        """
        self.inputs = inputs or []
        self.initializers = initializers or []
        self.nodes = nodes or []


def test_check_model() -> None:
    """Verify that check model behaves as expected."""
    # Valid model
    n1 = MockNode("Add", ["x", "y"], ["z"])
    g1 = MockGraph(inputs=[MockTensor("float", [1], is_initializer=False)], nodes=[n1])
    g1.inputs[0].name = "x"

    i1 = MockTensor("float", [1], is_initializer=True)
    i1.name = "y"
    g1.initializers.append(i1)

    m1 = MockModel(opset_import=[MockDomain("ai.onnx")], graph=g1)

    assert check_model(m1) is True

    # Invalid models
    m2 = MockModel(ir_version=1)
    with pytest.raises(ValidationError):
        check_model(m2)


def test_check_model_async() -> None:
    """Verify that check model async behaves as expected."""
    m1 = MockModel(
        opset_import=[MockDomain("ai.onnx")],
        graph=MockGraph(
            inputs=[MockTensor("float", [1])],
            nodes=[MockNode("Add", ["x", "y"], ["z"])],
        ),
    )
    m1.graph.inputs[0].name = "x"
    i1 = MockTensor("float", [1], is_initializer=True)
    i1.name = "y"
    m1.graph.initializers.append(i1)

    assert asyncio.run(check_model_async(m1)) is True


def test_check_op_specific() -> None:
    """Verify that check op specific behaves as expected."""
    ctx = ValidationContext()

    # Add
    n1 = MockNode("Add", ["x"], ["z"])
    _check_op_specific(n1, ctx)
    assert "Add requires 2 inputs" in ctx.errors
    ctx.errors.clear()

    # Conv
    n2 = MockNode("Conv", ["x"], ["z"])
    _check_op_specific(n2, ctx)
    assert "Conv requires at least 2 inputs" in ctx.errors
    ctx.errors.clear()

    n3 = MockNode("Conv", ["x", "w"], ["z"], {"pads": [1, 2, 3]})
    _check_op_specific(n3, ctx)
    assert "Conv pads must be 2 * spatial_dims" in ctx.errors
    ctx.errors.clear()

    # Control flow
    n4 = MockNode("If", ["x"], ["z"])
    _check_op_specific(n4, ctx)
    assert "If requires subgraph attributes" in ctx.errors
    ctx.errors.clear()

    # TreeEnsemble
    n5 = MockNode("TreeEnsembleClassifier", ["x"], ["z"], {"nodes_treeids": [1]})
    _check_op_specific(n5, ctx)
    assert "TreeEnsembleClassifier missing attributes" in ctx.errors
    ctx.errors.clear()


def test_check_model_missing_fields() -> None:
    """Verify that check model missing fields behaves as expected."""
    # missing opset_import
    m1 = MockModel(producer_name=123)
    with pytest.raises(ValidationError) as e:
        check_model(m1)
    assert "Invalid producer_name, opset_import missing" in str(e.value)

    # duplicate opset_import domain
    m2 = MockModel(opset_import=[MockDomain("ai.onnx"), MockDomain("ai.onnx")], graph=MockGraph())
    with pytest.raises(ValidationError) as e:
        check_model(m2)
    assert "Duplicate domain ai.onnx" in str(e.value)

    # duplicate input
    i = MockTensor("float", [1])
    i.name = "dup"
    m3 = MockModel(opset_import=[MockDomain("ai.onnx")], graph=MockGraph(inputs=[i, i]))
    with pytest.raises(ValidationError) as e:
        check_model(m3)
    assert "Duplicate input dup" in str(e.value)

    # duplicate initializer
    m4 = MockModel(opset_import=[MockDomain("ai.onnx")], graph=MockGraph(initializers=[i, i]))
    with pytest.raises(ValidationError) as e:
        check_model(m4)
    assert "Duplicate initializer dup" in str(e.value)

    # duplicate output
    n = MockNode("Add", ["x", "y"], ["z", "z"])
    m5 = MockModel(opset_import=[MockDomain("ai.onnx")], graph=MockGraph(nodes=[n]))
    with pytest.raises(ValidationError) as e:
        check_model(m5)
    assert "Duplicate node output z" in str(e.value)

    # dangling input
    n2 = MockNode("Add", ["x", "missing"], ["z"])
    m6 = MockModel(opset_import=[MockDomain("ai.onnx")], graph=MockGraph(nodes=[n2]))
    with pytest.raises(ValidationError) as e:
        check_model(m6)
    assert "Dangling input x" in str(e.value)
