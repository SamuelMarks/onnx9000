import pytest
from onnx9000.core.exceptions import UnsupportedOpError
from onnx9000.core.registry import OperatorRegistry


def test_registry_registration() -> None:
    reg = OperatorRegistry()

    @reg.register_op("Add")
    def mock_add() -> str:
        return "added"

    assert reg.get_op("Add")() == "added"


def test_registry_unsupported() -> None:
    reg = OperatorRegistry()
    with pytest.raises(UnsupportedOpError):
        reg.get_op("MissingOp")


def test_registry_duplicate() -> None:
    reg = OperatorRegistry()

    @reg.register_op("Add")
    def mock_add() -> None:
        assert True

    with pytest.raises(ValueError):

        @reg.register_op("Add")
        def mock_add_again() -> None:
            assert True


def test_registry_new_domain() -> None:
    from onnx9000.core.registry import OperatorRegistry

    reg = OperatorRegistry()

    @reg.register_op("TestNewDomain", domain="custom.domain")
    def my_op() -> None:
        pass

    assert "custom.domain" in reg._domains


def test_global_register_op() -> None:
    from onnx9000.core.registry import global_registry, register_op

    @register_op("GlobalMockOp", domain="global")
    def global_mock() -> None:
        pass

    assert "global.GlobalMockOp" in global_registry._registry
