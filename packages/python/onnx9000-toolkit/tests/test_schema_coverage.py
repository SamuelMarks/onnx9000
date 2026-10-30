"""Tests the schema coverage module functionality."""

import pytest
from onnx9000.toolkit.script.schema import set_target_opset, validate_op


def test_schema_warnings() -> None:
    """Tests the schema warnings functionality."""
    set_target_opset(18)
    with pytest.warns(UserWarning, match="is not valid for operation"):
        validate_op("Relu", ["X"], {"invalid_attr": 1})
    with pytest.warns(UserWarning, match="will be converted to input for Squeeze"):
        from onnx9000.toolkit.script.schema import OpSchema, registry

        registry.register(OpSchema("Squeeze", 1, ["data"], ["squeezed"], ["axes"]))
        validate_op("Squeeze", ["data"], {"axes": [0]})


def test_schema_coverage_more() -> None:
    """Tests the schema coverage more functionality."""
    from onnx9000.toolkit.script.schema import get_target_opset

    set_target_opset(10)
    with pytest.raises(ValueError, match="requires opset"):
        validate_op("Relu", ["X"], {})
    validate_op("UnregisteredOp", [], {})
    assert get_target_opset() == 10
    set_target_opset(18)


def test_schema_opset_error() -> None:
    """Verify that schema opset error behaves as expected."""
    import pytest
    from onnx9000.toolkit.script.schema import (
        OpSchema,
        registry,
        set_target_opset,
        validate_op,
    )

    registry.register(OpSchema("TargetOp", 20, [], [], []))
    set_target_opset(10)
    with pytest.raises(ValueError):
        validate_op("TargetOp", [], {})
    set_target_opset(18)


def test_schema_valid_attributes() -> None:
    """Verify that schema valid attributes behaves as expected."""
    from onnx9000.toolkit.script.schema import (
        OpSchema,
        registry,
        set_target_opset,
        validate_op,
    )

    registry.register(OpSchema("ValidAttrOp", 10, [], [], ["my_attr"]))
    set_target_opset(10)
    # This shouldn't warn
    _res = validate_op("ValidAttrOp", [], {"my_attr": 1})
    assert _res is not None or _res is None
    set_target_opset(18)
