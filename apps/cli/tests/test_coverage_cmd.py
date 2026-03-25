"""Tests for the coverage command."""

import argparse
import json
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest

from onnx9000_cli.coverage import (
    get_pypi_info,
    generate_framework_snapshots,
    clone_and_parse_onnx_spec,
    get_onnx9000_ops,
    generate_markdown_table,
    generate_summary_table,
    count_supported_framework_objects,
    update_compliance_md,
    update_coverage_cmd,
)


def test_get_pypi_info():
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'{"info": {"version": "2.0.0", "requires_python": ">=3.8"}}'
        )
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        assert get_pypi_info("testpkg") == ("2.0.0", "3.8")

        mock_response.read.return_value = b'{"info": {"version": "2.0.0", "requires_python": null}}'
        assert get_pypi_info("testpkg") == ("2.0.0", None)

        mock_urlopen.side_effect = Exception("Network error")
        assert get_pypi_info("testpkg") == ("unknown", None)


def test_generate_framework_snapshots(tmpdir):
    snapshots_dir = str(tmpdir)

    def mock_get_pypi(pkg):
        if pkg == "caffe":
            return "unknown", None
        if pkg == "cntk":
            return "1.0.0", "3.6"  # triggers fallback and installation
        if pkg == "tensorflow":
            return "1.0.0", "3.10"  # triggers fallback but pyenv has it
        if pkg == "torch":
            return "1.0.0", "3.9"  # triggers complete pyenv fallback failure
        return "1.0.0", None

    def mock_subp_run(args, **kwargs):
        cmd = args[0]
        if cmd == "uv":
            if len(args) > 1 and args[1] == "venv":
                venv_dir = args[-1]
                os.makedirs(venv_dir, exist_ok=True)
                # First uv venv try fails to trigger pyenv
                if "--python" in args and args[3] in ["3.6", "3.10", "3.9"]:
                    raise subprocess.CalledProcessError(1, "uv")
        elif cmd == "pyenv":
            if args[1] == "versions":
                # Mock pyenv having 3.10 installed but not 3.6
                mock_res = MagicMock()
                mock_res.stdout = "3.10.1\n3.10.2\n"
                return mock_res
            elif args[1] == "install":
                if "3.9" in args:
                    raise subprocess.CalledProcessError(1, "pyenv install")
                return MagicMock()
            elif args[1] == "prefix":
                mock_res = MagicMock()
                mock_res.stdout = "/mock/pyenv/prefix"
                return mock_res

        # dumper.py output mock
        if len(args) >= 4 and args[1].endswith("dumper.py"):
            fw = args[2]
            out_path = args[3]
            if fw == "torch":
                raise subprocess.CalledProcessError(1, "python")  # mock failure at the end
            data = {"version": "1.0.0", "objects": ["obj1"]}
            with open(out_path, "w") as f:
                json.dump(data, f)
        return MagicMock()

    with (
        patch("onnx9000_cli.coverage.get_pypi_info", side_effect=mock_get_pypi),
        patch("subprocess.run", side_effect=mock_subp_run),
    ):
        # Pre-create a snapshot for 'onnx' to test the cache hit
        with open(os.path.join(snapshots_dir, "onnx-1.0.0.json"), "w") as f:
            json.dump({"version": "1.0.0", "objects": ["onnx_obj"]}, f)

        # Pre-create a corrupt snapshot for 'jax'
        with open(os.path.join(snapshots_dir, "jax-1.0.0.json"), "w") as f:
            f.write("corrupted_json")

        # Mock os.name for Windows branch coverage
        with patch("os.name", "nt"):
            results = generate_framework_snapshots(snapshots_dir)

        assert "onnx" in results
        assert results["onnx"]["objects"] == ["onnx_obj"]  # cache hit

        assert "caffe" in results
        assert results["caffe"]["version"] == "Not Installed"  # unknown pypi

        assert "jax" in results
        assert results["jax"]["version"] == "Not Installed"  # corrupted cache handling

        assert "tensorflow" in results
        assert results["tensorflow"]["objects"] == ["obj1"]  # normal venv run

        assert "torch" in results
        assert results["torch"]["version"] == "Not Installed"  # dumper returned Not Installed

        assert "cntk" in results
        assert results["cntk"]["version"] == "1.0.0"  # successfully installed and dumped


def test_count_supported_framework_objects(tmpdir):
    """Test counting supported framework objects."""
    with (
        patch("os.path.abspath", return_value=str(tmpdir)),
        patch("os.path.dirname", return_value=str(tmpdir)),
        patch.dict(
            "sys.modules",
            {"onnx9000.converters": MagicMock(__file__=os.path.join(tmpdir, "__init__.py"))},
        ),
    ):
        # Test non-existent path
        assert count_supported_framework_objects("unknown") == 0

        # Test tensorflow
        tf_dir = os.path.join(tmpdir, "tf")
        os.makedirs(tf_dir, exist_ok=True)
        with open(os.path.join(tf_dir, "test.py"), "w") as f:
            f.write("def _map_test(): pass\n")
        assert count_supported_framework_objects("tensorflow") == 1

        # Test paddle
        paddle_dir = os.path.join(tmpdir, "paddle")
        os.makedirs(paddle_dir, exist_ok=True)
        with open(os.path.join(paddle_dir, "test.py"), "w") as f:
            f.write("def _map_test(): pass\n")
        assert count_supported_framework_objects("paddle") == 1

        # Test torch
        torch_dir = os.path.join(tmpdir, "frontend")
        os.makedirs(torch_dir, exist_ok=True)
        with open(os.path.join(torch_dir, "test.py"), "w") as f:
            f.write("class Module: pass\nclass MyLayer(Module): pass\n")
        assert count_supported_framework_objects("torch") == 1

        # Test keras
        keras_file = os.path.join(tf_dir, "keras_layers.py")
        with open(keras_file, "w") as f:
            f.write("def _map_test(): pass\ndef _map_test2(): pass\n")
        assert count_supported_framework_objects("keras") == 2

        # Test jax
        jax_dir = os.path.join(tmpdir, "jax")
        os.makedirs(jax_dir, exist_ok=True)
        with open(os.path.join(jax_dir, "test.py"), "w") as f:
            f.write("def _map_test(): pass\n")
        assert count_supported_framework_objects("jax") == 1

        # Test coremltools
        mltools_dir = os.path.join(tmpdir, "mltools")
        os.makedirs(mltools_dir, exist_ok=True)
        with open(os.path.join(mltools_dir, "coreml.py"), "w") as f:
            f.write("def _map_test(): pass\n")
        assert count_supported_framework_objects("coremltools") == 1

        # Test new frameworks using _count_funcs
        sklearn_dir = os.path.join(tmpdir, "sklearn")
        os.makedirs(sklearn_dir, exist_ok=True)
        with open(os.path.join(sklearn_dir, "test.py"), "w") as f:
            f.write("def convert_test(): pass\n")
        assert count_supported_framework_objects("sklearn") == 1

        with open(os.path.join(mltools_dir, "xgboost.py"), "w") as f:
            f.write("def parse_xgboost_test(): pass\n")
        assert count_supported_framework_objects("xgboost") == 1

        with open(os.path.join(mltools_dir, "lightgbm.py"), "w") as f:
            f.write("def parse_lightgbm_test(): pass\n")
        assert count_supported_framework_objects("lightgbm") == 1

        with open(os.path.join(mltools_dir, "catboost.py"), "w") as f:
            f.write("def parse_catboost_test(): pass\n")
        assert count_supported_framework_objects("catboost") == 1

        with open(os.path.join(mltools_dir, "sparkml.py"), "w") as f:
            f.write("def parse_sparkml_test(): pass\n")
        assert count_supported_framework_objects("pyspark") == 1

        with open(os.path.join(mltools_dir, "h2o.py"), "w") as f:
            f.write("def parse_h2o(): pass\n")
        assert count_supported_framework_objects("h2o") == 1

        with open(os.path.join(mltools_dir, "libsvm.py"), "w") as f:
            f.write("def parse_libsvm(): pass\n")
        assert count_supported_framework_objects("libsvm") == 1

        safetensors_dir = os.path.join(tmpdir, "safetensors")
        os.makedirs(safetensors_dir, exist_ok=True)
        with open(os.path.join(safetensors_dir, "test.py"), "w") as f:
            f.write("def load_safetensor(): pass\n")
        assert count_supported_framework_objects("safetensors") == 1

    with (
        patch("os.path.abspath", return_value=os.path.join(str(tmpdir), "onnx9000-onnx2gguf")),
        patch("os.path.exists", side_effect=[True, True]),
    ):  # One for converters_dir, one for gguf_dir
        assert count_supported_framework_objects("gguf") == 2

    with (
        patch("os.path.abspath", return_value=os.path.join(str(tmpdir), "onnx9000-onnx2gguf")),
        patch("os.path.exists", side_effect=[True, False]),
    ):  # One for converters_dir, one for gguf_dir
        assert count_supported_framework_objects("gguf") == 0


def test_count_supported_framework_objects_exceptions(tmpdir):
    """Test exception paths in counting supported framework objects."""
    with (
        patch("os.path.abspath", return_value=os.path.join(str(tmpdir), "does_not_exist")),
        patch("builtins.__import__", side_effect=ImportError("Mock error")),
    ):
        # Test converters dir missing
        assert count_supported_framework_objects("tensorflow") == 0

    with (
        patch("os.path.abspath", return_value=str(tmpdir)),
        patch("builtins.__import__", side_effect=ImportError("Mock error")),
    ):
        # Test missing file error inside valid directory
        os.makedirs(os.path.join(tmpdir, "tf"), exist_ok=True)
        with open(os.path.join(tmpdir, "tf", "test.py"), "w") as f:
            f.write("invalid syntax === \n")
        # Syntax error should be caught and return 0
        assert count_supported_framework_objects("tensorflow") == 0

        # Test torch exception
        os.makedirs(os.path.join(tmpdir, "frontend"), exist_ok=True)
        with open(os.path.join(tmpdir, "frontend", "test.py"), "w") as f:
            f.write("invalid syntax === \n")
        assert count_supported_framework_objects("torch") == 0

        # Test _count_funcs exception
        os.makedirs(os.path.join(tmpdir, "sklearn"), exist_ok=True)
        with open(os.path.join(tmpdir, "sklearn", "test.py"), "w") as f:
            f.write("invalid syntax === \n")
        assert count_supported_framework_objects("sklearn") == 0

        # Test fallback return
        assert count_supported_framework_objects("unknown_framework_xyz") == 0


def test_generate_summary_table():
    frameworks = {
        "onnx": {"version": "1.0", "objects": ["a"]},
        "torch": {"version": "2.0", "objects": ["a", "b"]},
    }
    onnx_data = {"commit": "hash123", "operators": ["Abs", "Add"]}
    onnx9000_ops = ["abs"]

    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=1):
        md = generate_summary_table(frameworks, onnx_data, onnx9000_ops)
        assert "## Summary" in md
        assert "50.00%" in md
        assert "| Torch | 1 | 2 | 50.00% |" in md


def test_generate_summary_table_unknown():
    frameworks = {"torch": {"version": "2.0", "objects": []}}
    onnx_data = {"commit": "hash123", "operators": []}
    onnx9000_ops = []

    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=1):
        md = generate_summary_table(frameworks, onnx_data, onnx9000_ops)
        assert "| Torch | 1 | Unknown | N/A |" in md


def test_update_compliance_md(tmpdir):
    """Test update_compliance_md logic."""
    md_path = os.path.join(tmpdir, "specs", "ONNX01_COMPLIANCE.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)

    with (
        patch("os.path.abspath", return_value=md_path),
        patch("os.getcwd", return_value=str(tmpdir)),
    ):
        # Test file missing
        update_compliance_md("dummy")

        # Test with Description
        with open(md_path, "w") as f:
            f.write("## Description\nSome text.\n\n### Next")
        update_compliance_md("SUMMARY1")
        with open(md_path, "r") as f:
            content = f.read()
            assert (
                "<!-- COVERAGE_SUMMARY_START -->\nSUMMARY1\n<!-- COVERAGE_SUMMARY_END -->"
                in content
            )

        # Test replacement
        update_compliance_md("SUMMARY2")
        with open(md_path, "r") as f:
            content = f.read()
            assert "SUMMARY2" in content
            assert "SUMMARY1" not in content

        # Test fallback
        with open(md_path, "w") as f:
            f.write("Just some text")
        update_compliance_md("SUMMARY3")
        with open(md_path, "r") as f:
            content = f.read()
            assert "SUMMARY3" in content


def test_clone_and_parse_onnx_spec(tmpdir):
    """Test clone_and_parse_onnx_spec."""
    with patch("subprocess.run") as mock_run:
        # Mock git rev-parse output
        mock_run_result = MagicMock()
        mock_run_result.stdout = "mock_hash"
        mock_run.return_value = mock_run_result

        # Create dummy temp directory with file
        with patch("tempfile.TemporaryDirectory") as mock_temp:
            mock_temp.return_value.__enter__.return_value = str(tmpdir)

            os.makedirs(os.path.join(tmpdir, "docs"), exist_ok=True)
            with open(os.path.join(tmpdir, "docs", "Operators.md"), "w") as f:
                f.write('## <a name="Abs"></a><a name="abs">**Abs**</a>\n')
                f.write('## <a name="Add"></a>**Add**\n')
                f.write("Some other text\n")
                f.write(
                    '## <a name="Cast"></a><a name="cast">**Cast**\n'
                )  # Malformed but > catches it maybe?
                f.write('## <a name="Div"></a>>Div\n')
                f.write('|<a href="#Mul">Mul</a>|something|\n')
                f.write('## <a name="invalid"></a>\n')

            result = clone_and_parse_onnx_spec()
            assert result["commit"] == "mock_hash"
            assert "Abs" in result["operators"]
            assert "Add" in result["operators"]
            assert "Cast" in result["operators"]
            assert "Div" in result["operators"]
            assert "Mul" in result["operators"]
            assert "invalid" not in result["operators"]  # Doesn't start with uppercase


def test_clone_and_parse_onnx_spec_fail():
    """Test clone_and_parse_onnx_spec on subprocess error."""
    import subprocess

    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")):
        result = clone_and_parse_onnx_spec()
        assert result["commit"] == "unknown"
        assert result["operators"] == []


def test_get_onnx9000_ops(tmpdir):
    """Test get_onnx9000_ops real case."""
    ops_dir = os.path.join(tmpdir, "ops")
    os.makedirs(ops_dir, exist_ok=True)
    with open(os.path.join(ops_dir, "test.py"), "w") as f:
        f.write("def abs(): pass\ndef _hidden(): pass\ndef record_op(): pass\n")

    real_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "onnx9000.core":
            raise ImportError("Mock error")
        return real_import(name, *args, **kwargs)

    with (
        patch("os.path.abspath", return_value=ops_dir),
        patch("builtins.__import__", side_effect=mock_import),
    ):
        ops = get_onnx9000_ops()
        assert isinstance(ops, list)
        assert "abs" in ops
        assert "_hidden" not in ops
        assert "record_op" not in ops


def test_get_onnx9000_ops_import_error(tmpdir):
    """Test get_onnx9000_ops import error."""
    real_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "onnx9000.core":
            raise ImportError("Mock error")
        return real_import(name, *args, **kwargs)

    def mock_import_attr(name, *args, **kwargs):
        if name == "onnx9000.core":
            m = MagicMock()
            del m.ops
            return m
        return real_import(name, *args, **kwargs)

    with (
        patch("os.path.abspath", return_value=os.path.join(str(tmpdir), "does_not_exist")),
        patch("builtins.__import__", side_effect=mock_import),
    ):
        ops = get_onnx9000_ops()
        assert ops == []

    with (
        patch("os.path.abspath", return_value=os.path.join(str(tmpdir), "does_not_exist")),
        patch("builtins.__import__", side_effect=mock_import_attr),
    ):
        ops = get_onnx9000_ops()
        assert ops == []

    ops_dir = os.path.join(tmpdir, "ops")
    os.makedirs(ops_dir, exist_ok=True)
    with open(os.path.join(ops_dir, "test.py"), "w") as f:
        f.write("invalid syntax === \n")

    with (
        patch("os.path.abspath", return_value=ops_dir),
        patch("builtins.__import__", side_effect=mock_import),
    ):
        ops = get_onnx9000_ops()
        assert ops == []


def test_generate_markdown_table():
    frameworks = {"onnx": {"version": "1.0", "objects": ["a"]}}
    onnx_data = {"commit": "hash123", "operators": ["Abs", "Add", "Sub", "Div"]}
    onnx9000_ops = ["abs", "sub"]  # Now tests exact match since _ is stripped

    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=1):
        md = generate_markdown_table(frameworks, onnx_data, onnx9000_ops)
        assert "Supported Frameworks Coverage" in md
        assert "hash123" in md
        assert "2/4 (50.00%)" in md
        assert "| Abs | ✅ |" in md
        assert "| Add | ❌ |" in md
        assert "| Sub | ✅ |" in md
        assert "| Div | ❌ |" in md
        assert "| onnx | 1.0 |" in md


def test_generate_markdown_table_empty():
    frameworks = {}
    onnx_data = {"commit": "unknown", "operators": []}
    onnx9000_ops = []

    with patch("onnx9000_cli.coverage.count_supported_framework_objects", return_value=1):
        md = generate_markdown_table(frameworks, onnx_data, onnx9000_ops)
        assert "Supported Frameworks Coverage" in md
        assert "0/0" not in md  # Division by zero protected


def test_update_coverage_cmd(tmpdir):
    with (
        patch("onnx9000_cli.coverage.generate_framework_snapshots") as mock_fw,
        patch("onnx9000_cli.coverage.clone_and_parse_onnx_spec") as mock_onnx,
        patch("onnx9000_cli.coverage.get_onnx9000_ops") as mock_ops,
        patch("os.getcwd", return_value=str(tmpdir)),
    ):
        mock_fw.return_value = {"onnx": {"version": "1.0", "objects": ["a"]}}
        mock_onnx.return_value = {"commit": "hash123", "operators": ["Abs", "Add"]}
        mock_ops.return_value = ["abs"]

        import argparse

        args = argparse.Namespace()
        update_coverage_cmd(args)

        assert os.path.exists(os.path.join(tmpdir, "snapshots", "onnx-hash123.json"))
        assert os.path.exists(os.path.join(tmpdir, "SUPPORTED_PER_FRAMEWORK.md"))
