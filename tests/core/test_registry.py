import pytest
from onnx9000.core.registry import OperatorRegistry
from onnx9000.core.exceptions import UnsupportedOpError


def test_registry_already_registered():
    """Test ValueError is raised when registering an operator twice."""
    registry = OperatorRegistry()

    @registry.register("Add")
    def gen_add():
        return "add"

    assert registry.get_generator("Add")() == "add"
    with pytest.raises(ValueError, match="Operator Add is already registered."):

        @registry.register("Add")
        def gen_add2():
            return "add2"


def test_registry_get_unsupported():
    """Test UnsupportedOpError is raised for unknown operators."""
    registry = OperatorRegistry()
    with pytest.raises(UnsupportedOpError):
        registry.get_generator("UnknownOp")


def test_registry_load_plugin():
    """Test dynamic loading of a plugin module."""
    registry = OperatorRegistry()
    registry.load_plugin("math")


def test_registry_invalid_domain():
    """Test ValueError is raised when registering to an unknown domain."""
    registry = OperatorRegistry()
    with pytest.raises(ValueError, match="Domain 'unknown' is not registered."):

        @registry.register("Add", domain="unknown")
        def gen_add():
            return "add"
