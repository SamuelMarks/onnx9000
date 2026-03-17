"""Tests the registry module functionality."""

import pytest
from onnx9000.core.exceptions import UnsupportedOpError
from onnx9000.core.registry import OperatorRegistry


def test_registry_registration() -> None:
    """Tests the registry registration functionality."""
    reg = OperatorRegistry()

    @reg.register_op("Add")
    def mock_add() -> str:
        """Tests the mock add functionality."""
        return "added"

    assert reg.get_op("Add")() == "added"


def test_registry_unsupported() -> None:
    """Tests the registry unsupported functionality."""
    reg = OperatorRegistry()
    with pytest.raises(UnsupportedOpError):
        reg.get_op("MissingOp")


def test_registry_duplicate() -> None:
    """Tests the registry duplicate functionality."""
    reg = OperatorRegistry()

    @reg.register_op("Add")
    def mock_add() -> None:
        """Tests the mock add functionality."""
        assert True

    with pytest.raises(ValueError):

        @reg.register_op("Add")
        def mock_add_again() -> None:
            """Tests the mock add again functionality."""
            assert True


def test_registry_new_domain() -> None:
    """Tests the registry new domain functionality."""
    from onnx9000.core.registry import OperatorRegistry

    reg = OperatorRegistry()

    @reg.register_op("TestNewDomain", domain="custom.domain")
    def my_op() -> None:
        """Tests the my op functionality."""
        pass

    assert "custom.domain" in reg._domains


def test_global_register_op() -> None:
    """Tests the global register op functionality."""
    from onnx9000.core.registry import global_registry, register_op

    @register_op("GlobalMockOp", domain="global")
    def global_mock() -> None:
        """Tests the global mock functionality."""
        pass

    assert "global.GlobalMockOp" in global_registry._registry
