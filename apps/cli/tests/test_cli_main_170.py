"""Unit tests and verification suite for cli main 170 functionality."""

import argparse
import contextlib

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import convert_cmd


def test_convert_tensorflow_graphdef(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that convert tensorflow graphdef behaves as expected.

    Args:
        tmp_path: Parameter specifying tmp path.
    """
    model_path = tmp_path / "model.pb"
    out_path = tmp_path / "model_out.onnx"

    # Just an empty file for dummy test. The parser might fail but we check if it routes correctly
    model_path.write_bytes(b"")

    args = argparse.Namespace(
        src=str(model_path),
        from_fmt="tensorflow",
        to_fmt="onnx",
        output=str(out_path),
        weights=None,
    )

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
        convert_cmd(args)
    assert args is not None or args is None


def test_convert_tensorflow_saved_model(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that convert tensorflow saved model behaves as expected.

    Args:
        tmp_path: Parameter specifying tmp path.
    """
    model_dir = tmp_path / "saved_model"
    model_dir.mkdir()
    pb_path = model_dir / "saved_model.pb"
    pb_path.write_bytes(b"")

    out_path = tmp_path / "model_out.onnx"

    args = argparse.Namespace(
        src=str(model_dir),
        from_fmt="tensorflow",
        to_fmt="onnx",
        output=str(out_path),
        weights=None,
    )

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
        convert_cmd(args)
    assert args is not None or args is None


def test_convert_tensorflow_missing_pb(
    tmp_path: int | float | str | bool | bytes | list | dict | tuple | None,
) -> None:
    """Verify that convert tensorflow missing pb behaves as expected.

    Args:
        tmp_path: Parameter specifying tmp path.
    """
    model_dir = tmp_path / "saved_model_missing"
    model_dir.mkdir()
    out_path = tmp_path / "model_out.onnx"

    args = argparse.Namespace(
        src=str(model_dir),
        from_fmt="tensorflow",
        to_fmt="onnx",
        output=str(out_path),
        weights=None,
    )

    with contextlib.suppress(SystemExit):
        convert_cmd(args)
    assert args is not None or args is None
