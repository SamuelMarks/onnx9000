"""Tests the registry module functionality."""

import pytest
from onnx9000.core.exceptions import UnsupportedOpError
from onnx9000.core.registry import OperatorRegistry


def test_registry_registration() -> None:
    """Tests the registry registration functionality."""
    reg = OperatorRegistry()

    @reg.register_op("", "Add")
    def mock_add() -> str:
        """Test the mock add functionality."""
        return "added"

    assert reg.get_op("", "Add")() == "added"


def test_registry_unsupported() -> None:
    """Tests the registry unsupported functionality."""
    reg = OperatorRegistry()
    with pytest.raises(UnsupportedOpError):
        reg.get_op("", "MissingOp")


def test_registry_duplicate() -> None:
    """Tests the registry duplicate functionality."""
    reg = OperatorRegistry()

    @reg.register_op("", "Add")
    def mock_add() -> None:
        """Provides functional implementation."""
        return None

    mock_add()

    @reg.register_op("", "Add")
    def mock_add_again() -> None:
        """Provides functional implementation."""
        return None

    mock_add_again()

    assert reg.get_op("", "Add") is mock_add_again


def test_registry_new_domain() -> None:
    """Tests the registry new domain functionality."""
    from onnx9000.core.registry import OperatorRegistry

    reg = OperatorRegistry()

    @reg.register_op("custom.domain", "TestNewDomain")
    def my_op() -> None:
        """Provides functional implementation."""
        return None

    my_op()

    assert "custom.domain" in reg._domains


def test_global_register_op() -> None:
    """Tests the global register op functionality."""
    from onnx9000.core.registry import global_registry, register_op

    @register_op("global", "GlobalMockOp")
    def global_mock() -> None:
        """Provides functional implementation."""
        return None

    global_mock()

    assert ("global", "GlobalMockOp", None) in global_registry._registry


def test_registry_pattern_validations() -> None:
    """Dynamically inspect the Python ast and fail if ANY execution mapping is hardcoded instead of using the @register_op framework."""
    import ast
    from pathlib import Path

    # Check ops module
    ops_dir = Path(__file__).parent.parent / "src" / "onnx9000" / "core" / "ops"

    for py_file in ops_dir.glob("**/*.py"):
        if py_file.name == "torch_auto.py":
            continue  # massive auto gen file, skips fine mapping structure

        with open(py_file, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                # Must have a register_op decorator
                has_register = False
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call) and getattr(dec.func, "id", None) == "register_op":
                        has_register = True
                        break

                # We assert that every public function inside ops is registered via register_op pattern
                if node.name not in [
                    "record_op",
                    "concat_from_sequence",
                    "split_to_sequence",
                    "make_dim",
                    "get_tensor_data_as_list",
                ]:
                    assert has_register, (
                        f"Function {node.name} in {py_file} missing @register_op decorator!"
                    )
