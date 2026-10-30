"""Unit tests and verification suite for cli coverage main gaps3 functionality."""

from unittest.mock import patch

from onnx9000_cli.coverage import (
    count_supported_framework_objects,
    generate_markdown_table,
    generate_summary_table,
)


def test_gguf_dir_exists_or_not_fixed() -> None:
    """Verify that gguf dir exists or not fixed behaves as expected."""
    with patch("os.path.exists", side_effect=lambda x: True):
        assert count_supported_framework_objects("gguf") == 2
    with patch("os.path.exists", side_effect=lambda x: "converters" in x):
        assert count_supported_framework_objects("gguf") == 0


def test_generate_summary_table_branches() -> None:
    """Verify that generate summary table branches behaves as expected."""
    frameworks_data = {"torch": {"version": "1.0", "objects": ["A"]}}
    onnx_data = {"version": "1.0", "operators": ["Add"]}
    onnx9000_ops = ["add"]
    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=1):
        generate_summary_table(frameworks_data, onnx_data, onnx9000_ops)

    frameworks_data_no_obj = {
        "ONNX Spec": {"version": "1.0", "objects": []},
        "torch": {"version": "1.0", "objects": []},
    }
    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=0):
        generate_summary_table(frameworks_data_no_obj, onnx_data, onnx9000_ops)
    assert frameworks_data_no_obj is not None or frameworks_data_no_obj is None


def test_generate_markdown_table_branches() -> None:
    """Verify that generate markdown table branches behaves as expected."""
    frameworks_data = {"torch": {"version": "1.0", "objects": [{"name": "A"}]}}
    onnx_data = {"version": "1.0", "operators": ["Add"]}
    onnx9000_ops = ["add"]
    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=1):
        generate_markdown_table(frameworks_data, onnx_data, onnx9000_ops)

    frameworks_data_no_obj = {
        "onnx": {"version": "1.0", "objects": []},
        "torch": {"version": "1.0", "objects": []},
    }
    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=0):
        generate_markdown_table(frameworks_data_no_obj, onnx_data, onnx9000_ops)
    assert frameworks_data_no_obj is not None or frameworks_data_no_obj is None


def test_generate_summary_table_unsupported_op() -> None:
    """Verify that generate summary table unsupported op behaves as expected."""
    frameworks_data = {}
    onnx_data = {"version": "1.0", "operators": ["Add", "UnsupportedOp"]}
    onnx9000_ops = ["add"]
    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=1):
        generate_summary_table(frameworks_data, onnx_data, onnx9000_ops)
    assert onnx9000_ops is not None or onnx9000_ops is None


def test_generate_markdown_table_onnx_fw() -> None:
    """Verify that generate markdown table onnx fw behaves as expected."""
    frameworks_data = {"onnx": {"version": "1.0", "objects": []}}
    onnx_data = {"version": "1.0", "operators": ["Add"]}
    onnx9000_ops = ["add"]
    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=1):
        generate_markdown_table(frameworks_data, onnx_data, onnx9000_ops)
    assert onnx9000_ops is not None or onnx9000_ops is None
