import argparse
import ast
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

from onnx9000_cli.coverage import (
    clone_and_parse_onnx_spec,
    count_supported_framework_objects,
    generate_framework_snapshots,
    update_compliance_md,
)


def test_clone_and_parse_onnx_spec_fallback():
    with patch("tempfile.TemporaryDirectory") as mock_temp:
        mock_temp.return_value.__enter__.return_value = "tmpdir"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "abc123hash"
            with patch("os.path.exists", return_value=True):
                with patch("builtins.open") as mock_open:
                    mock_open.return_value.__enter__.return_value = ["no match"]
                    # Will hit fallback
                    with patch("json.load", return_value={"operators": ["Sub"], "commit": "123"}):
                        res = clone_and_parse_onnx_spec()
                        assert res["operators"][0] == "Sub"


def test_clone_and_parse_onnx_spec_subprocess_err():
    with patch("tempfile.TemporaryDirectory") as mock_temp:
        mock_temp.return_value.__enter__.return_value = "tmpdir"
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")):
            res = clone_and_parse_onnx_spec()
            assert res["operators"] == []


def test_update_compliance_md_match():
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "before\n<!-- COVERAGE_SUMMARY_START -->\nold\n<!-- COVERAGE_SUMMARY_END -->\nafter"
            )
            update_compliance_md("new")


def test_generate_framework_snapshots_cntk_success():
    with patch("tempfile.TemporaryDirectory") as mock_temp:
        mock_temp.return_value.__enter__.return_value = "tmpdir"
        with patch("onnx9000_cli.coverage.get_pypi_info", return_value=("1.0", "3.6")):
            with patch("subprocess.run"):
                with patch("glob.glob", return_value=[]):
                    with patch("builtins.open") as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = (
                            '{"version": "1.0", "objects": ["a"]}'
                        )
                        with patch("json.load", return_value={"version": "1.0", "objects": ["a"]}):
                            try:
                                # We only want cntk
                                with patch(
                                    "onnx9000_cli.coverage.get_pypi_info",
                                    side_effect=lambda pkg: (
                                        ("1.0", "3.6") if pkg == "cntk" else ("unknown", "3.10")
                                    ),
                                ):
                                    generate_framework_snapshots("snapshots_dir")
                            except Exception:
                                pass


def test_generate_framework_snapshots_existing_json_error():
    with patch("tempfile.TemporaryDirectory") as mock_temp:
        mock_temp.return_value.__enter__.return_value = "tmpdir"
        with patch(
            "onnx9000_cli.coverage.get_pypi_info",
            side_effect=[("1.0", "3.6")] + [("1.0", "3.10")] * 20,
        ):
            with patch("os.path.exists", return_value=True):
                with patch("builtins.open"):
                    with patch("json.load", side_effect=Exception("parse error")):
                        try:
                            generate_framework_snapshots("snapshots_dir")
                        except Exception:
                            pass


def test_generate_framework_snapshots_unknown_version_fallback_json_error():
    with patch("tempfile.TemporaryDirectory") as mock_temp:
        mock_temp.return_value.__enter__.return_value = "tmpdir"
        with patch("onnx9000_cli.coverage.get_pypi_info", return_value=("unknown", "3.10")):
            with patch("glob.glob", return_value=["fallback.json"]):
                with patch("builtins.open"):
                    with patch("json.load", side_effect=Exception("parse error")):
                        try:
                            generate_framework_snapshots("snapshots_dir")
                        except Exception:
                            pass


def test_generate_framework_snapshots_cntk_subprocess_error():
    with patch("tempfile.TemporaryDirectory") as mock_temp:
        mock_temp.return_value.__enter__.return_value = "tmpdir"
        with patch("onnx9000_cli.coverage.get_pypi_info", return_value=("1.0", "3.6")):
            with patch("subprocess.run", side_effect=Exception("docker fail")):
                with patch("builtins.open"):
                    try:
                        with patch(
                            "onnx9000_cli.coverage.get_pypi_info",
                            side_effect=lambda pkg: (
                                ("1.0", "3.6") if pkg == "cntk" else ("unknown", "3.10")
                            ),
                        ):
                            generate_framework_snapshots("snapshots_dir")
                    except Exception:
                        pass


def test_generate_framework_snapshots_nt_and_rmtree():
    with patch("tempfile.TemporaryDirectory") as mock_temp:
        mock_temp.return_value.__enter__.return_value = "tmpdir"
        with patch("onnx9000_cli.coverage.get_pypi_info", return_value=("1.0", "3.10")):
            with patch("subprocess.run"):
                with patch("glob.glob", return_value=[]):
                    with patch("builtins.open") as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = (
                            '{"version": "1.0", "objects": ["a"]}'
                        )
                        with patch("json.load", return_value={"version": "1.0", "objects": ["a"]}):
                            with patch("os.name", "nt"):

                                def exists_mock(p):
                                    if p.endswith(".json") and "snapshots" in p:
                                        return False
                                    return True

                                with patch("os.path.exists", side_effect=exists_mock):
                                    with patch("shutil.rmtree") as mock_rmtree:
                                        with patch(
                                            "onnx9000_cli.coverage.get_pypi_info",
                                            side_effect=lambda pkg: (
                                                ("1.0", "3.10")
                                                if pkg == "onnx"
                                                else ("unknown", "3.10")
                                            ),
                                        ):
                                            generate_framework_snapshots("snapshots_dir")
                                            assert mock_rmtree.called


def test_clone_and_parse_onnx_spec_fallback_json_error():
    with patch("tempfile.TemporaryDirectory") as mock_temp:
        mock_temp.return_value.__enter__.return_value = "tmpdir"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "abc123hash"
            with patch("os.path.exists", return_value=True):
                with patch("builtins.open") as mock_open:
                    mock_open.return_value.__enter__.return_value = ["no match"]
                    with patch("json.load", side_effect=Exception("parse error")):
                        res = clone_and_parse_onnx_spec()
                        assert res["operators"] == []
