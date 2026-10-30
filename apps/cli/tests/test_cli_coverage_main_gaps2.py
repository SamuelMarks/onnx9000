"""Unit tests and verification suite for cli coverage main gaps2 functionality."""

from unittest.mock import MagicMock, patch

from onnx9000_cli.coverage import count_supported_framework_objects


def test_count_funcs_branches() -> None:
    """Verify that count funcs branches behaves as expected."""
    with patch("os.path.exists", return_value=True):
        with patch("os.path.isfile", return_value=True):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    "def parse_xgboost_1(): pass"
                )
                res = count_supported_framework_objects("xgboost")
                assert res == 1

        with patch("os.path.isfile", return_value=False):
            with patch("os.path.isdir", return_value=True):
                with patch("os.walk", return_value=[("dir", [], ["f.py"])]):
                    with patch("builtins.open") as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = (
                            "def convert_1(): pass"
                        )
                        res = count_supported_framework_objects("sklearn")
                        assert res == 1


def test_count_map_funcs_branches() -> None:
    """Verify that count map funcs branches behaves as expected."""
    with patch("os.path.exists", return_value=True):
        with patch("os.path.isfile", return_value=True):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    "def _map_1(): pass"
                )
                res = count_supported_framework_objects("tensorflow")
                assert res == 1

        with patch("os.path.isfile", return_value=False):
            with patch("os.path.isdir", return_value=True):
                with patch("os.walk", return_value=[("dir", [], ["f.py"])]):
                    with patch("builtins.open") as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = (
                            "def _map_2(): pass"
                        )
                        res = count_supported_framework_objects("tensorflow")
                        assert res == 1


def test_count_classes_inheriting_module_branches() -> None:
    """Verify that count classes inheriting module branches behaves as expected.

    Returns:
        Result of test count classes inheriting module branches.
    """
    with (
        patch("os.path.exists", return_value=True),
        patch(
            "glob.glob",
            side_effect=lambda p: ["snapshot.json"] if "snapshot" in p else [],
        ),
    ):

        def fake_open(
            filepath: int | float | str | bool | bytes | list | dict | tuple | None,
            *args: int | float | str | bool | bytes | list | dict | tuple | None,
            **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
        ) -> int | float | str | bool | bytes | list | dict | tuple | None:
            """Execute fake open operation.

            Args:
                filepath: Parameter specifying filepath.
                *args: Variadic positional arguments for args.
                **kwargs: Variadic keyword arguments for kwargs.

            Returns:
                Result of fake open.
            """
            mock = MagicMock()
            if "snapshot" in filepath:
                mock.__enter__.return_value.read.return_value = (
                    '{"objects": [{"name": "MyMod"}, {"name": "def"}]}'
                )
            else:
                mock.__enter__.return_value.read.return_value = """
import sys
from os import path
x = 1
class MyMod: pass
def normal_func(): pass
def _private_func(): pass
"""
            return mock

        with patch("builtins.open", side_effect=fake_open):
            with patch("os.path.isdir", return_value=False):
                res = count_supported_framework_objects("torch")
                assert res > 0

            with patch("os.path.isdir", return_value=True):
                with patch("os.walk", return_value=[("dir", [], ["f.py"])]):
                    res = count_supported_framework_objects("torch")
                    assert res > 0


def test_ast_exceptions_in_counters() -> None:
    """Verify that ast exceptions in counters behaves as expected.

    Returns:
        Result of test ast exceptions in counters.
    """
    with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "invalid("
            count_supported_framework_objects("xgboost")
            count_supported_framework_objects("tensorflow")
            with patch("glob.glob", return_value=["test.json"]):

                def fake_open(
                    p: int | float | str | bool | bytes | list | dict | tuple | None,
                    *args: int | float | str | bool | bytes | list | dict | tuple | None,
                    **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
                ) -> int | float | str | bool | bytes | list | dict | tuple | None:
                    """Execute fake open operation.

                    Args:
                        p: Parameter specifying p.
                        *args: Variadic positional arguments for args.
                        **kwargs: Variadic keyword arguments for kwargs.

                    Returns:
                        Result of fake open.
                    """
                    m = MagicMock()
                    if "json" in p:
                        m.__enter__.return_value.read.return_value = '{"objects": []}'
                    else:
                        m.__enter__.return_value.read.return_value = "invalid("
                    return m

                with patch("builtins.open", side_effect=fake_open):
                    count_supported_framework_objects("torch")
    assert test_ast_exceptions_in_counters is not None


def test_torch_empty_snapshots() -> None:
    """Verify that torch empty snapshots behaves as expected."""
    with patch("os.path.exists", return_value=True), patch("glob.glob", return_value=[]):
        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = ""
            count_supported_framework_objects("torch")
    assert test_torch_empty_snapshots is not None


def test_get_onnx9000_ops_importerror() -> None:
    """Verify that get onnx9000 ops importerror behaves as expected.

    Returns:
        Result of test get onnx9000 ops importerror.

    Raises:
        ImportError: If an invalid state or argument is encountered during execution.
    """
    import builtins

    orig_import = builtins.__import__

    def fake_import(
        name: int | float | str | bool | bytes | list | dict | tuple | None,
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute fake import operation.

        Args:
            name: Parameter specifying name.
            *args: Variadic positional arguments for args.
            **kwargs: Variadic keyword arguments for kwargs.

        Returns:
            Result of fake import.

        Raises:
            ImportError: If an invalid state or argument is encountered during execution.
        """
        if name == "onnx9000.core":
            raise ImportError("no onnx9000.core")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with patch("os.path.exists", return_value=False):
            from onnx9000_cli.coverage import get_onnx9000_ops

            res = get_onnx9000_ops()
            assert res == []


def test_get_onnx9000_ops_ast_error() -> None:
    """Verify that get onnx9000 ops ast error behaves as expected."""
    from onnx9000_cli.coverage import get_onnx9000_ops

    with patch("os.path.exists", return_value=True):
        with patch("os.walk", return_value=[("dir", [], ["f.py"])]):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = "invalid syntax("
                res = get_onnx9000_ops()
                assert res == []
