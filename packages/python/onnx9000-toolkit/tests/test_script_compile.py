"""Unit tests for compiling ONNX 9000 script-decorated models.

This module verifies that Python functions decorated with `@script` can be
correctly parsed and compiled into ONNX 9000 executable or graph representations.
It also ensures that appropriate errors are raised when compilation is attempted
on codebases lacking the required decorator configurations.
"""

import os
import tempfile

import pytest
from onnx9000.toolkit.script import parse_and_compile


def test_parse_and_compile() -> None:
    """Verify that a valid script-decorated function compiles successfully.

    This test writes a temporary Python module containing a function annotated
    with the `@script` decorator and uses the `parse_and_compile` utility to compile it.
    It asserts that the compilation process completes successfully and returns a
    valid, non-None model.

    Returns:
        None
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("""
from onnx9000.toolkit.script import script, op

@script
def test_model(x):
    '''Compute the rectified linear unit (ReLU) activation of the input.

    Args:
        x (Var): The input variable or tensor on which to apply the Relu operation.

    Returns:
        Var: The output variable containing the element-wise maximum of x and 0.
    '''
    return op.Relu(x)
""")
        f_path = f.name

    try:
        model = parse_and_compile(f_path)
        assert model is not None
    finally:
        os.remove(f_path)


def test_parse_and_compile_no_script() -> None:
    """Verify compilation fails with ValueError when no @script decorator is present.

    This test writes a temporary Python module containing a function without any
    decorators. It expects `parse_and_compile` to fail with a specific ValueError indicating
    that no script-decorated function was found.

    Returns:
        None
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("""
def test_model(x):
    '''An identity function used to verify compilation failure behaviors.

    Args:
        x (Any): The input object to be returned unmodified.

    Returns:
        Any: The same input object as passed in.
    '''
    return x
""")
        f_path = f.name

    try:
        with pytest.raises(ValueError, match="No @script decorated function found"):
            parse_and_compile(f_path)
    finally:
        os.remove(f_path)
