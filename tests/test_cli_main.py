"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.cli.main import main
import argparse
import sys
from unittest.mock import patch


def test_cli_compile_success():
    """Provides semantic functionality and verification."""
    with (
        patch("argparse.ArgumentParser.parse_args") as mock_args,
        patch("onnx9000.cli.main.core_compile") as mock_compile,
    ):
        mock_args.return_value = argparse.Namespace(
            command="compile", model="good.onnx", target="cpp"
        )
        main()
        mock_compile.assert_called()


def test_cli_compile_failure():
    """Provides semantic functionality and verification."""
    with (
        patch("argparse.ArgumentParser.parse_args") as mock_args,
        patch("onnx9000.cli.main.core_compile", side_effect=Exception("mock err")),
        patch("sys.exit") as mock_exit,
    ):
        mock_args.return_value = argparse.Namespace(
            command="compile", model="bad.onnx", target="cpp"
        )
        main()
        mock_exit.assert_called_with(1)
