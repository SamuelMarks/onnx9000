from unittest.mock import patch, MagicMock
from onnx9000_cli.coverage import (
    count_supported_framework_objects,
    generate_summary_table,
    generate_markdown_table,
)


def test_gguf_dir_exists_or_not_fixed():
    with patch("os.path.exists", side_effect=lambda x: True):
        assert count_supported_framework_objects("gguf") == 2
    with patch("os.path.exists", side_effect=lambda x: True if "converters" in x else False):
        assert count_supported_framework_objects("gguf") == 0


def test_generate_summary_table_branches():
    frameworks_data = {"torch": {"version": "1.0", "objects": ["A"]}}
    onnx_data = {"version": "1.0", "operators": ["Add"]}
    onnx9000_ops = ["add"]
    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=1):
        res = generate_summary_table(frameworks_data, onnx_data, onnx9000_ops)

    frameworks_data_no_obj = {
        "ONNX Spec": {"version": "1.0", "objects": []},
        "torch": {"version": "1.0", "objects": []},
    }
    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=0):
        res2 = generate_summary_table(frameworks_data_no_obj, onnx_data, onnx9000_ops)


def test_generate_markdown_table_branches():
    frameworks_data = {"torch": {"version": "1.0", "objects": [{"name": "A"}]}}
    onnx_data = {"version": "1.0", "operators": ["Add"]}
    onnx9000_ops = ["add"]
    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=1):
        res = generate_markdown_table(frameworks_data, onnx_data, onnx9000_ops)

    frameworks_data_no_obj = {
        "onnx": {"version": "1.0", "objects": []},
        "torch": {"version": "1.0", "objects": []},
    }
    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=0):
        res2 = generate_markdown_table(frameworks_data_no_obj, onnx_data, onnx9000_ops)


def test_generate_summary_table_unsupported_op():
    frameworks_data = {}
    onnx_data = {"version": "1.0", "operators": ["Add", "UnsupportedOp"]}
    onnx9000_ops = ["add"]
    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=1):
        generate_summary_table(frameworks_data, onnx_data, onnx9000_ops)


def test_generate_markdown_table_onnx_fw():
    frameworks_data = {"onnx": {"version": "1.0", "objects": []}}
    onnx_data = {"version": "1.0", "operators": ["Add"]}
    onnx9000_ops = ["add"]
    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=1):
        generate_markdown_table(frameworks_data, onnx_data, onnx9000_ops)
