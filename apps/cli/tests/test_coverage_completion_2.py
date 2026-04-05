import pytest
from unittest.mock import patch, MagicMock
from onnx9000_cli.coverage import count_supported_framework_objects, clone_and_parse_onnx_spec


def test_clone_and_get_operators_fallback():
    with patch("tempfile.TemporaryDirectory") as mock_temp_dir:
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/fake_dir"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "fake_hash"

            with patch("os.path.exists", return_value=True):
                with patch("builtins.open") as mock_open:
                    mock_file = MagicMock()
                    mock_file.__enter__.return_value = []

                    mock_json_file = MagicMock()
                    mock_json_file.__enter__.return_value.read.return_value = '["Add", "Sub"]'

                    mock_open.side_effect = [mock_file, mock_json_file]

                    with patch("json.load", return_value=["Add", "Sub"]):
                        res = clone_and_parse_onnx_spec()
                        assert res["commit"] == "fake_hash"


def test_get_onnx9000_ops_exception():
    from onnx9000_cli.coverage import get_onnx9000_ops

    with patch("os.walk", return_value=[("/fake", [], ["test.py"])]):
        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.side_effect = Exception(
                "Parse failed"
            )
            ops = get_onnx9000_ops()
            assert ops == []


def test_count_supported_framework_objects_exceptions():
    with patch("os.path.isdir", return_value=True):
        with patch("os.walk", return_value=[("/fake", [], ["test.py"])]):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.side_effect = Exception(
                    "Parse failed"
                )
                count = count_supported_framework_objects("sklearn")

    with patch("os.path.isdir", return_value=True):
        with patch("glob.glob", return_value=["test.json"]):
            with patch("builtins.open") as mock_open:
                with patch("json.load", return_value={"objects": []}):
                    mock_open.return_value.__enter__.return_value.read.side_effect = Exception(
                        "Parse failed"
                    )
                    count_supported_framework_objects("pytorch")

    with patch("os.path.isdir", return_value=True):
        with patch("os.walk", return_value=[("/fake", [], ["test.py"])]):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.side_effect = Exception(
                    "Parse failed"
                )
                count_supported_framework_objects("tensorflow")


def test_generate_summary_table():
    from onnx9000_cli.coverage import generate_summary_table

    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=100):
        res = generate_summary_table(
            {"pytorch": {"objects": ["o1" for _ in range(100)]}, "onnx": {}},
            {"operators": ["Add", "Sub"]},
            ["Add", "Sub"],
        )
        assert "| Pytorch | 100 | 100 | 100.00% |" in res


def test_generate_markdown_table():
    from onnx9000_cli.coverage import generate_markdown_table

    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=1):
        res = generate_markdown_table(
            {"pytorch": {"objects": [{"name": "Add"}]}}, {"operators": ["Add"]}, ["Add"]
        )
        assert "| Pytorch | 1 | 1 | 100.00% |" in res


def test_clone_and_get_operators_success():
    with patch("tempfile.TemporaryDirectory") as mock_temp_dir:
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/fake_dir"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "fake_hash"

            with patch("os.path.exists", return_value=True):
                with patch("builtins.open") as mock_open:
                    mock_file = MagicMock()
                    mock_file.__enter__.return_value = [
                        '## <a name="Abs"></a>**Abs**\n',
                        '## <a name="Add"></a><a name="Add"></a>**Add**\n',
                        '## <a name="Sub"></a>**Sub**\n',
                    ]

                    mock_open.return_value = mock_file

                    res = clone_and_parse_onnx_spec()
                    assert "Abs" in res["operators"]
                    assert "Sub" in res["operators"]
                    assert "Add" not in res["operators"]


def test_count_classes_inheriting_module_ast_nodes_2():
    from onnx9000_cli.coverage import count_supported_framework_objects

    with patch("os.path.isdir", return_value=True):
        with patch("glob.glob", return_value=["test.json"]):
            with patch("builtins.open") as mock_open:
                mock_file = MagicMock()
                mock_file.__enter__.return_value.read.return_value = (
                    '{"objects": [{"name":"Class1"}, {"name":"pass"}, {"name":"if"}]}'
                )
                mock_open.return_value = mock_file

                with patch(
                    "json.load",
                    return_value={
                        "objects": [{"name": "Class1"}, {"name": "pass"}, {"name": "if"}]
                    },
                ):
                    with patch("os.walk", return_value=[("/fake", [], ["test.py"])]):
                        with patch("builtins.open") as mock_open2:
                            mock_open2.return_value.__enter__.return_value.read.return_value = (
                                "class Class1: pass"
                            )
                            count = count_supported_framework_objects("pytorch")


def test_count_map_funcs_ast_nodes_2():
    from onnx9000_cli.coverage import count_supported_framework_objects

    with patch("os.path.isdir", return_value=True):
        with patch("os.walk", return_value=[("/fake", [], ["test.py"])]):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    "def _map_func(): pass"
                )
                count = count_supported_framework_objects("tensorflow")
                assert count == 1
                count2 = count_supported_framework_objects("paddle")
                assert count2 == 1


def test_coverage_ast_exceptions():
    from onnx9000_cli.coverage import get_onnx9000_ops, count_supported_framework_objects

    with patch("os.path.exists", return_value=True):
        with patch("os.walk", return_value=[("/fake", [], ["test.py"])]):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.side_effect = Exception(
                    "Parse failed"
                )
                ops = get_onnx9000_ops()
                assert ops == []

    with patch("os.path.isdir", return_value=True):
        with patch("os.walk", return_value=[("/fake", [], ["test.py"])]):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.side_effect = Exception(
                    "Parse failed"
                )
                count = count_supported_framework_objects("sklearn")
                assert count == 0

    with patch("os.path.isdir", return_value=True):
        with patch("os.walk", return_value=[("/fake", [], ["test.py"])]):
            with patch("builtins.open") as mock_open:
                with patch("json.load", return_value={"objects": []}):
                    mock_open.return_value.__enter__.return_value.read.side_effect = Exception(
                        "Parse failed"
                    )
                    count = count_supported_framework_objects("torch")
                    assert count == 0

    with patch("os.path.isdir", return_value=True):
        with patch("os.walk", return_value=[("/fake", [], ["test.py"])]):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.side_effect = Exception(
                    "Parse failed"
                )
                count = count_supported_framework_objects("tensorflow")
                assert count == 0


def test_count_classes_inheriting_module_valid_ast():
    from onnx9000_cli.coverage import count_supported_framework_objects

    with patch("os.path.isdir", return_value=True):
        with patch("os.walk", return_value=[("/fake", [], ["test.py"])]):
            with patch("glob.glob", return_value=["fake.json"]):
                with patch(
                    "json.load",
                    return_value={
                        "objects": [
                            {"name": "fake1"},
                            {"name": "fake2"},
                            {"name": "fake3"},
                            {"name": "fake4"},
                        ]
                    },
                ):

                    def my_open(f, *args, **kwargs):
                        m = MagicMock()
                        if f.endswith(".json"):
                            m.__enter__.return_value.read.return_value = '{"objects": [{"name": "fake1"}, {"name": "fake2"}, {"name": "fake3"}, {"name": "fake4"}]}'
                        else:
                            m.__enter__.return_value.read.return_value = "class fake1: pass\ndef fake2(): pass\nfake3 = 1\nfrom math import fake4\nimport math as fake4"
                        return m

                    with patch("builtins.open", side_effect=my_open):
                        count = count_supported_framework_objects("torch")
                        assert count >= 3


def test_count_supported_framework_objects_misc():
    from onnx9000_cli.coverage import count_supported_framework_objects

    count = count_supported_framework_objects("coreml")
    count = count_supported_framework_objects("tflite")
    count = count_supported_framework_objects("gguf")


def test_get_onnx9000_ops_success():
    from onnx9000_cli.coverage import get_onnx9000_ops

    with patch("os.path.exists", return_value=True):
        with patch("os.walk", return_value=[("/fake", [], ["test.py"])]):
            with patch("builtins.open") as mock_open:
                mock_file = MagicMock()
                mock_file.__enter__.return_value.read.return_value = (
                    'record_op("Add")\nrecord_op("Sub")'
                )
                mock_open.return_value = mock_file
                ops = get_onnx9000_ops()
                assert "add" in ops
                assert "sub" in ops


def test_count_supported_framework_objects_jax_success():
    from onnx9000_cli.coverage import count_supported_framework_objects

    with patch("os.path.isdir", return_value=True):
        with patch("os.walk", return_value=[("/fake", [], ["test.py"])]):
            with patch("builtins.open") as mock_open:
                mock_file = MagicMock()
                mock_file.__enter__.return_value.read.return_value = (
                    "def convert_func(): pass\nclass Other: pass\n"
                )
                mock_open.return_value = mock_file
                # jax uses prefix 'jax_' in _count_funcs
                count = count_supported_framework_objects("sklearn")
                assert count == 1


def test_isfile_branches():
    from onnx9000_cli.coverage import count_supported_framework_objects

    # 510
    with patch("os.path.isfile", return_value=True):
        with patch("os.path.isdir", return_value=False):
            with patch("builtins.open") as mock_open:
                mock_file = MagicMock()
                mock_file.__enter__.return_value.read.return_value = (
                    "def convert_func(): pass\nclass Other: pass\n"
                )
                mock_open.return_value = mock_file
                count = count_supported_framework_objects("sklearn")
                assert count == 1

    # 637
    with patch("os.path.isfile", return_value=True):
        with patch("os.path.isdir", return_value=False):
            with patch("builtins.open") as mock_open:
                mock_file = MagicMock()
                mock_file.__enter__.return_value.read.return_value = (
                    "def _map_func(): pass\nclass Other: pass\n"
                )
                mock_open.return_value = mock_file
                count = count_supported_framework_objects("tensorflow")
                assert count == 1

    # 745-746
    from onnx9000_cli.coverage import generate_summary_table

    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=0):
        res = generate_summary_table(
            {"pytorch": {"objects": []}, "onnx": {}}, {"operators": ["Add", "Sub"]}, ["Add", "Sub"]
        )
        assert "| Pytorch | 0 | Unknown | N/A |" in res
