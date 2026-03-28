"""Tests for the coverage command."""

import argparse
import json
import os
import subprocess
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
    """Verify that get_pypi_info correctly retrieves and parses package info from PyPI."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'{"info": {"version": "2.0.0", "requires_python": ">=3.8"}}'
        )
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        assert get_pypi_info("testpkg") == ("2.0.0", "3.11")
        assert get_pypi_info("cntk") == ("2.0.0", "3.6")
        assert get_pypi_info("mxnet") == ("2.0.0", "3.8")
        assert get_pypi_info("caffe") == ("2.0.0", "3.8")

        mock_response.read.return_value = b'{"info": {"version": "2.0.0", "requires_python": null}}'
        assert get_pypi_info("testpkg") == ("2.0.0", "3.11")

        mock_urlopen.side_effect = Exception("Network error")
        assert get_pypi_info("testpkg") == ("unknown", "3.11")


def test_generate_framework_snapshots(tmpdir):
    """Test the generation of framework snapshots across different installation methods."""
    snapshots_dir = str(tmpdir)

    def mock_get_pypi(pkg):
        """Tests the PyPI package info retrieval functionality."""
        if pkg == "mxnet":
            return "1.0.0", "3.8"  # Trigger docker block
        if pkg == "catboost":
            return "unknown", None  # Test unknown fallback
        if pkg == "cntk":
            return "1.0.0", "3.6"  # legacy
        if pkg == "tensorflow":
            return "1.0.0", "3.10"
        if pkg == "torch":
            return "1.0.0", "3.9"
        if pkg == "flax":
            return "1.0.0", "3.7"
        if pkg == "coremltools":
            return "2.0.0", "3.10"
        return "1.0.0", None

    def mock_subp_run(args, **kwargs):
        """Tests the subprocess execution functionality for framework installers."""
        if isinstance(args, str):
            if "docker run" in args:
                if "mxnet" in args:
                    raise Exception("Docker failure")
                fw = args.split()[-2].strip("\"'")
                out_path = os.path.join(snapshots_dir, f"{fw}-1.0.0.json")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump({"version": "1.0.0", "objects": ["obj1"]}, f)
                return MagicMock()
            cmd = args.split()[0]
        else:
            cmd = args[0]
        if cmd == "uv":
            if len(args) > 1 and args[1] == "venv":
                venv_dir = args[-1]
                os.makedirs(venv_dir, exist_ok=True)
                if "--python" in args and args[3] in ["3.6", "3.10", "3.9", "3.7"]:
                    raise subprocess.CalledProcessError(1, "uv")
            if len(args) > 2 and args[2] == "install":
                if "coremltools" in args[-1]:
                    raise subprocess.CalledProcessError(1, "pip install")
        elif cmd == "pyenv":
            if args[1] == "versions":
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

        if len(args) >= 4 and args[1].endswith("dumper.py"):
            fw = args[2]
            out_path = args[3]
            if fw == "torch":
                raise subprocess.CalledProcessError(1, "python")
            data = {"version": "1.0.0", "objects": ["obj1"]}
            with open(out_path, "w") as f:
                json.dump(data, f)
        return MagicMock()

    with (
        patch("onnx9000_cli.coverage.get_pypi_info", side_effect=mock_get_pypi),
        patch("onnx9000_cli.coverage.subprocess.run", side_effect=mock_subp_run),
    ):
        with open(os.path.join(snapshots_dir, "onnx-1.0.0.json"), "w") as f:
            json.dump({"version": "1.0.0", "objects": ["onnx_obj"]}, f)

        with open(os.path.join(snapshots_dir, "jax-1.0.0.json"), "w") as f:
            f.write("corrupted_json")

        with open(os.path.join(snapshots_dir, "catboost-0.8.0.json"), "w") as f:
            json.dump({"version": "0.8.0", "objects": ["catboost_obj"]}, f)

        with open(os.path.join(snapshots_dir, "coremltools-1.0.0.json"), "w") as f:
            json.dump({"version": "1.0.0", "objects": ["coreml_obj"]}, f)

        with patch("os.name", "nt"):
            results = generate_framework_snapshots(snapshots_dir)

        assert "onnx" in results
        assert results["mxnet"]["version"] == "Not Installed"
        assert results["catboost"]["version"] == "0.8.0"
        assert results["coremltools"]["version"] == "1.0.0"


def test_clone_and_parse_onnx_spec_errors(tmpdir):
    """Test error conditions in clone_and_parse_onnx_spec."""
    # Test subprocess error for coverage of line 448
    with patch(
        "onnx9000_cli.coverage.subprocess.run", side_effect=subprocess.CalledProcessError(1, "git")
    ):
        res = clone_and_parse_onnx_spec()
        assert res["commit"] == "unknown"


def test_get_onnx9000_ops_more(tmpdir):
    """Test more edge cases for get_onnx9000_ops."""
    # Test ops_dir does not exist for coverage of line 471
    with (
        patch("os.path.abspath", return_value="/non/existent"),
        patch.dict("sys.modules", {"onnx9000.core": None}),
    ):
        assert get_onnx9000_ops() == []

    # Test ast parse error for coverage of line 488
    ops_dir = os.path.join(tmpdir, "ops_err")
    os.makedirs(ops_dir, exist_ok=True)
    with open(os.path.join(ops_dir, "err.py"), "w") as f:
        f.write("invalid syntax !!!")

    with (
        patch("os.path.abspath", return_value=ops_dir),
        patch.dict("sys.modules", {"onnx9000.core": None}),
    ):
        assert get_onnx9000_ops() == []


def test_count_supported_framework_objects_more(tmpdir):
    """Test more edge cases for count_supported_framework_objects."""
    # Test converters_dir does not exist for coverage of line 512
    with (
        patch("os.path.abspath", return_value="/non/existent"),
        patch.dict("sys.modules", {"onnx9000": None}),
    ):
        assert count_supported_framework_objects("tensorflow") == 0

    # Test exceptions in sub-counters for coverage of lines 543, 571, 604
    conv_dir = os.path.join(tmpdir, "conv_err")
    os.makedirs(conv_dir, exist_ok=True)
    os.makedirs(os.path.join(conv_dir, "tf"), exist_ok=True)
    os.makedirs(os.path.join(conv_dir, "frontend"), exist_ok=True)
    os.makedirs(os.path.join(conv_dir, "sklearn"), exist_ok=True)

    with open(os.path.join(conv_dir, "tf", "err.py"), "w") as f:
        f.write("invalid syntax")
    with open(os.path.join(conv_dir, "frontend", "err.py"), "w") as f:
        f.write("invalid syntax")
    with open(os.path.join(conv_dir, "sklearn", "err.py"), "w") as f:
        f.write("invalid syntax")

    with (
        patch("os.path.abspath", return_value=conv_dir),
        patch.dict(
            "sys.modules",
            {
                "onnx9000": MagicMock(
                    converters=MagicMock(__file__=os.path.join(conv_dir, "__init__.py"))
                )
            },
        ),
    ):
        assert count_supported_framework_objects("tensorflow") == 0
        assert count_supported_framework_objects("torch") == 0
        assert count_supported_framework_objects("sklearn") == 0


def test_generate_markdown_table_coverage(tmpdir):
    """Test markdown table generation coverage for line 723-725."""
    frameworks = {}
    onnx_data = {"commit": "h", "operators": ["Abs"]}
    onnx9000_ops = ["abs"]
    md = generate_markdown_table(frameworks, onnx_data, onnx9000_ops)
    assert "1/1" in md


def test_update_compliance_md_fallback(tmpdir):
    """Test update_compliance_md fallback for line 797."""
    md_path = os.path.join(tmpdir, "COMPLIANCE.md")
    with open(md_path, "w") as f:
        f.write("Empty")

    with patch("os.path.abspath", return_value=md_path):
        update_compliance_md("SUM")
        with open(md_path, "r") as f:
            content = f.read()
            assert "SUM" in content


def test_update_coverage_cmd_real(tmpdir):
    """Test update_coverage_cmd with minimal mocks."""
    args = argparse.Namespace()
    with (
        patch("onnx9000_cli.coverage.generate_framework_snapshots", return_value={}),
        patch(
            "onnx9000_cli.coverage.clone_and_parse_onnx_spec",
            return_value={"commit": "h", "operators": []},
        ),
        patch("onnx9000_cli.coverage.get_onnx9000_ops", return_value=[]),
        patch("os.getcwd", return_value=str(tmpdir)),
        patch("onnx9000_cli.coverage.update_compliance_md"),
    ):
        update_coverage_cmd(args)
        assert os.path.exists(os.path.join(tmpdir, "SUPPORTED_PER_FRAMEWORK.md"))
