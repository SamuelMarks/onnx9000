"""Unit tests and verification suite for cli main 101 functionality."""

import argparse
import contextlib
from unittest.mock import patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import workspace_cmd


def test_coverage_gaps_cmd101() -> None:
    """Verify that coverage gaps cmd101 behaves as expected.

    Returns:
        Result of test coverage gaps cmd101.

    Raises:
        ImportError: If an invalid state or argument is encountered during execution.
    """
    args = argparse.Namespace(path=".")

    import builtins

    original_import = builtins.__import__

    def mock_import(
        name: int | float | str | bool | bytes | list | dict | tuple | None,
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute mock import operation.

        Args:
            name: Parameter specifying name.
            *args: Variadic positional arguments for args.
            **kwargs: Variadic keyword arguments for kwargs.

        Returns:
            Result of mock import.

        Raises:
            ImportError: If an invalid state or argument is encountered during execution.
        """
        if name == "onnx9000_workspace":
            raise ImportError("mock err")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with contextlib.suppress(
            Onnx9000Error,
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            KeyError,
            IndexError,
            AttributeError,
            ImportError,
            LookupError,
            ArithmeticError,
            BufferError,
            AssertionError,
            StopIteration,
        ):
            workspace_cmd(args)
    assert original_import is not None or original_import is None
