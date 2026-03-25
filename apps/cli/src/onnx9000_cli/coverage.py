"""Coverage tracking command for ONNX9000."""

import argparse
import ast
import importlib
import inspect
import json
import os
import subprocess
import tempfile
import urllib.request
from typing import Dict, List, Set, Any, Optional, Tuple

import urllib.error
import shutil
import re


def get_pypi_info(pkg_name: str) -> Tuple[str, Optional[str]]:
    """Fetch the latest version and python requirement of a package from PyPI.

    Returns:
        A tuple of (latest_version, required_python_version).
        Returns ('unknown', None) if the package is not found.
    """
    url = f"https://pypi.org/pypi/{pkg_name}/json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            version = data["info"]["version"]
            requires_python = data["info"].get("requires_python")
            py_ver = None
            if requires_python:
                match = re.search(r"(3\.\d+)", requires_python)
                if match:
                    py_ver = match.group(1)
            return version, py_ver
    except Exception:
        return "unknown", None


def generate_framework_snapshots(snapshots_dir: str) -> Dict[str, Dict[str, Any]]:
    """Generate API snapshots by querying PyPI and creating temporary venvs.

    Returns:
        A dictionary mapping framework names to their version and exposed objects.
    """
    frameworks = [
        "onnx",
        "torch",
        "tensorflow",
        "keras",
        "jax",
        "flax",
        "paddle",
        "coremltools",
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
        "pyspark",
        "h2o",
        "libsvm",
        "cntk",
        "mxnet",
        "caffe",
        "gguf",
        "safetensors",
    ]
    pkg_mapping = {
        "sklearn": "scikit-learn",
        "paddle": "paddlepaddle",
        "pyspark": "pyspark",
    }
    python_versions = {
        "cntk": "3.6",
        "mxnet": "3.8",
        "caffe": "3.8",
        "tensorflow": "3.10",
        "torch": "3.10",
    }
    default_python = "3.10"

    results = {}

    with tempfile.TemporaryDirectory() as base_tmp:
        dumper_path = os.path.join(base_tmp, "dumper.py")
        with open(dumper_path, "w", encoding="utf-8") as f:
            f.write(
                "import importlib, json, sys\n"
                "fw = sys.argv[1]\n"
                "out_path = sys.argv[2]\n"
                "try:\n"
                "    mod = importlib.import_module(fw)\n"
                "    version = getattr(mod, '__version__', 'unknown')\n"
                "    objects = mod.__all__ if hasattr(mod, '__all__') else [n for n in dir(mod) if not n.startswith('_')]\n"
                "    objects = [str(o) for o in objects if isinstance(o, str)]\n"
                "    with open(out_path, 'w', encoding='utf-8') as f:\n"
                "        json.dump({'version': version, 'objects': objects}, f)\n"
                "except Exception as e:\n"
                "    with open(out_path, 'w', encoding='utf-8') as f:\n"
                "        json.dump({'version': 'Not Installed', 'objects': []}, f)\n"
            )

        for fw in frameworks:
            pkg_name = pkg_mapping.get(fw, fw)
            print(f"Checking PyPI for {pkg_name}...")
            version, pypi_py_ver = get_pypi_info(pkg_name)

            snapshot_path = os.path.join(snapshots_dir, f"{fw}-{version}.json")
            if os.path.exists(snapshot_path):
                print(f"Snapshot already exists for {fw}=={version}. Skipping venv creation.")
                try:
                    with open(snapshot_path, "r", encoding="utf-8") as f:
                        results[fw] = json.load(f)
                except Exception:
                    results[fw] = {"version": "Not Installed", "objects": []}
            elif version == "unknown":
                print(f"Could not find {pkg_name} on PyPI. Skipping.")
                results[fw] = {"version": "Not Installed", "objects": []}
            else:
                py_ver = pypi_py_ver or python_versions.get(fw, default_python)
                venv_dir = os.path.join(base_tmp, f"venv_{fw}")

                print(f"Creating virtualenv for {fw}=={version} with Python {py_ver}...")
                try:
                    try:
                        subprocess.run(
                            ["uv", "venv", "--python", py_ver, venv_dir],
                            check=True,
                            capture_output=True,
                        )
                    except subprocess.CalledProcessError:
                        print(f"uv failed to find Python {py_ver}, falling back to pyenv...")
                        try:
                            pyenv_vers = subprocess.run(
                                ["pyenv", "versions", "--bare"],
                                capture_output=True,
                                text=True,
                                check=True,
                            )
                            installed = pyenv_vers.stdout.splitlines()
                            matching = [v for v in installed if v.startswith(py_ver)]
                            if not matching:
                                print(
                                    f"Installing Python {py_ver} via pyenv (this may take a while)..."
                                )
                                subprocess.run(
                                    ["pyenv", "install", "-s", py_ver],
                                    check=True,
                                    capture_output=True,
                                )
                                best_ver = py_ver
                            else:
                                best_ver = matching[-1]

                            pyenv_prefix = subprocess.run(
                                ["pyenv", "prefix", best_ver],
                                check=True,
                                capture_output=True,
                                text=True,
                            )
                            pyenv_python = os.path.join(
                                pyenv_prefix.stdout.strip(), "bin", "python"
                            )

                            subprocess.run(
                                ["uv", "venv", "--python", pyenv_python, venv_dir],
                                check=True,
                                capture_output=True,
                            )
                        except Exception as pyenv_err:
                            raise RuntimeError(f"pyenv fallback failed: {pyenv_err}") from pyenv_err

                    print(f"Installing {pkg_name}=={version}...")
                    subprocess.run(
                        ["uv", "pip", "install", "--python", venv_dir, f"{pkg_name}=={version}"],
                        check=True,
                        capture_output=True,
                    )

                    python_exe = os.path.join(venv_dir, "bin", "python")
                    if os.name == "nt":
                        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")

                    tmp_out = os.path.join(base_tmp, f"{fw}_out.json")
                    subprocess.run(
                        [python_exe, dumper_path, fw, tmp_out], check=True, capture_output=True
                    )

                    with open(tmp_out, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)

                    results[fw] = data
                    print(f"Successfully generated API snapshot for {fw}.")
                except Exception as e:
                    print(f"Failed to generate API snapshot for {fw}: {e}")
                    results[fw] = {"version": "Not Installed", "objects": []}
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        json.dump(results[fw], f, indent=2)
                finally:
                    if os.path.exists(venv_dir):
                        shutil.rmtree(venv_dir)

    return results


def clone_and_parse_onnx_spec() -> Dict[str, Any]:
    """Clone ONNX repository and parse Operators.md.

    Returns:
        A dictionary containing the commit hash and a list of ONNX operators.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/onnx/onnx.git", temp_dir],
                check=True,
                capture_output=True,
            )
            # get commit hash
            commit_hash_proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=temp_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            commit_hash = commit_hash_proc.stdout.strip()

            operators_md_path = os.path.join(temp_dir, "docs", "Operators.md")
            operators = []
            if os.path.exists(operators_md_path):
                import re

                op_pattern = re.compile(r'<a href="#([A-Za-z0-9_]+)">')
                with open(operators_md_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("## <a name="):
                            parts = line.split("**")
                            if len(parts) >= 3:
                                operators.append(parts[1])
                            else:
                                if ">" in line:
                                    op_name = line.split(">")[-1].strip().replace("*", "")
                                    if op_name:
                                        operators.append(op_name)
                        else:
                            match = op_pattern.search(line)
                            if match:
                                operators.append(match.group(1))

            # remove duplicates and clean up
            operators = list(sorted(set([op for op in operators if op and op[0].isupper()])))
            return {"commit": commit_hash, "operators": operators}
        except subprocess.CalledProcessError:
            return {"commit": "unknown", "operators": []}


def get_onnx9000_ops() -> List[str]:
    """Get the ops supported by ONNX9000.

    Returns:
        A list of supported ONNX operator names (in lowercase).
    """
    import ast

    try:
        from onnx9000.core import ops

        ops_dir = os.path.dirname(ops.__file__)
    except (ImportError, AttributeError):
        ops_dir = os.path.abspath(
            os.path.join(
                os.getcwd(), "packages", "python", "onnx9000-core", "src", "onnx9000", "core", "ops"
            )
        )

    if not os.path.exists(ops_dir):
        return []

    ops_list = []
    files = [
        os.path.join(root, f) for root, _, fs in os.walk(ops_dir) for f in fs if f.endswith(".py")
    ]

    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            try:
                tree = ast.parse(fp.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        name = node.name
                        if not name.startswith("_") and name != "record_op":
                            ops_list.append(name.lower().replace("_", ""))
            except Exception:
                pass
    return ops_list


def count_supported_framework_objects(fw_name: str) -> int:
    """Dynamically count the number of supported mapping functions/classes for a framework."""
    try:
        from onnx9000 import converters

        converters_dir = os.path.dirname(converters.__file__)
    except (ImportError, AttributeError):
        converters_dir = os.path.abspath(
            os.path.join(
                os.getcwd(),
                "packages",
                "python",
                "onnx9000-converters",
                "src",
                "onnx9000",
                "converters",
            )
        )

    if not os.path.exists(converters_dir):
        return 0

    def _count_map_funcs(path: str) -> int:
        count = 0
        files = []
        if os.path.isfile(path):
            files = [path]
        elif os.path.isdir(path):
            files = [
                os.path.join(root, f)
                for root, _, fs in os.walk(path)
                for f in fs
                if f.endswith(".py")
            ]

        for f in files:
            with open(f, "r", encoding="utf-8") as file:
                try:
                    tree = ast.parse(file.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name.startswith("_map_"):
                            count += 1
                except Exception:
                    pass
        return count

    def _count_classes_inheriting_module(path: str) -> int:
        count = 0
        if not os.path.exists(path):
            return 0
        files = [
            os.path.join(root, f) for root, _, fs in os.walk(path) for f in fs if f.endswith(".py")
        ]
        for f in files:
            with open(f, "r", encoding="utf-8") as file:
                try:
                    tree = ast.parse(file.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            for base in node.bases:
                                if isinstance(base, ast.Name) and base.id == "Module":
                                    count += 1
                except Exception:
                    pass
        return count

    def _count_funcs(path: str, prefix: str) -> int:
        count = 0
        files = []
        if os.path.isfile(path):
            files = [path]
        elif os.path.isdir(path):
            files = [
                os.path.join(root, f)
                for root, _, fs in os.walk(path)
                for f in fs
                if f.endswith(".py")
            ]

        for f in files:
            with open(f, "r", encoding="utf-8") as file:
                try:
                    tree = ast.parse(file.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name.startswith(prefix):
                            count += 1
                except Exception:
                    pass
        return count

    if fw_name == "tensorflow":
        return _count_map_funcs(os.path.join(converters_dir, "tf"))
    elif fw_name == "paddle":
        return _count_map_funcs(os.path.join(converters_dir, "paddle"))
    elif fw_name == "torch":
        return _count_classes_inheriting_module(os.path.join(converters_dir, "frontend"))
    elif fw_name == "keras":
        return _count_map_funcs(os.path.join(converters_dir, "tf", "keras_layers.py"))
    elif fw_name in ["jax", "flax"]:
        return _count_map_funcs(os.path.join(converters_dir, "jax"))
    elif fw_name == "coremltools":
        return _count_map_funcs(os.path.join(converters_dir, "mltools", "coreml.py"))
    elif fw_name == "sklearn":
        return _count_funcs(os.path.join(converters_dir, "sklearn"), "convert_")
    elif fw_name == "xgboost":
        return _count_funcs(os.path.join(converters_dir, "mltools", "xgboost.py"), "parse_xgboost_")
    elif fw_name == "lightgbm":
        return _count_funcs(
            os.path.join(converters_dir, "mltools", "lightgbm.py"), "parse_lightgbm_"
        )
    elif fw_name == "catboost":
        return _count_funcs(
            os.path.join(converters_dir, "mltools", "catboost.py"), "parse_catboost_"
        )
    elif fw_name == "pyspark":
        return _count_funcs(os.path.join(converters_dir, "mltools", "sparkml.py"), "parse_sparkml_")
    elif fw_name == "h2o":
        return _count_funcs(os.path.join(converters_dir, "mltools", "h2o.py"), "parse_h2o")
    elif fw_name == "libsvm":
        return _count_funcs(os.path.join(converters_dir, "mltools", "libsvm.py"), "parse_libsvm")
    elif fw_name == "safetensors":
        return _count_funcs(os.path.join(converters_dir, "safetensors"), "load_")
    elif fw_name == "gguf":
        gguf_dir = os.path.abspath(
            os.path.join(
                os.getcwd(),
                "packages",
                "python",
                "onnx9000-onnx2gguf",
                "src",
                "onnx9000",
                "onnx2gguf",
            )
        )
        if os.path.exists(gguf_dir):
            return 2
        return 0
    else:
        return 0


def generate_summary_table(
    frameworks_data: Dict[str, Any], onnx_data: Dict[str, Any], onnx9000_ops: List[str]
) -> str:
    """Generate the summary table.

    Args:
        frameworks_data: Data about installed python frameworks.
        onnx_data: Data extracted from ONNX spec.
        onnx9000_ops: List of ops supported by ONNX9000.

    Returns:
        The markdown string for the summary.
    """
    onnx_ops = [op.lower() for op in onnx_data.get("operators", [])]
    onnx_supported = 0
    onnx_total = len(onnx_ops)
    if onnx_total > 0:
        for op in onnx_ops:
            if op in onnx9000_ops:
                onnx_supported += 1

    lines = ["## Summary\n"]
    lines.append("| Target | Supported | Total | Percentage |")
    lines.append("|---|---|---|---|")

    onnx_pct = f"{(onnx_supported / onnx_total * 100):.2f}%" if onnx_total > 0 else "0.00%"
    lines.append(f"| ONNX Spec | {onnx_supported} | {onnx_total} | {onnx_pct} |")

    for fw, data in frameworks_data.items():
        if fw == "onnx":
            continue
        total = len(data.get("objects", []))
        supported = count_supported_framework_objects(fw)
        if total == 0:
            total_disp = "Unknown"
            pct = "N/A"
        else:
            total_disp = str(total)
            pct = f"{(min(supported, total) / total * 100):.2f}%"
        lines.append(f"| {fw.capitalize()} | {supported} | {total_disp} | {pct} |")
    return "\n".join(lines)


def generate_markdown_table(
    frameworks_data: Dict[str, Any], onnx_data: Dict[str, Any], onnx9000_ops: List[str]
) -> str:
    """Generate the markdown table for SUPPORTED_PER_FRAMEWORK.md.

    Args:
        frameworks_data: Data about installed python frameworks.
        onnx_data: Data extracted from ONNX spec.
        onnx9000_ops: List of ops supported by ONNX9000.

    Returns:
        The markdown string.
    """
    lines = ["# Supported Frameworks Coverage\n"]
    lines.append("This file tracks the level of support for various ML frameworks in ONNX9000.\n")

    lines.append(generate_summary_table(frameworks_data, onnx_data, onnx9000_ops))

    onnx_ops = [op.lower() for op in onnx_data.get("operators", [])]
    onnx_supported = 0
    onnx_total = len(onnx_ops)
    if onnx_total > 0:
        for op in onnx_ops:
            if op in onnx9000_ops:
                onnx_supported += 1
    onnx_pct = f"{(onnx_supported / onnx_total * 100):.2f}%" if onnx_total > 0 else "0.00%"

    lines.append("\n## ONNX Spec Coverage\n")
    if onnx_total > 0:
        lines.append(f"**Coverage:** {onnx_supported}/{onnx_total} ({onnx_pct})\n")
        commit_hash = onnx_data.get("commit")
        lines.append(
            f"**Commit:** [`{commit_hash}`](https://github.com/onnx/onnx/commit/{commit_hash})\n"
        )

    lines.append("\n## Framework Versions\n")
    lines.append("| Framework | Version |")
    lines.append("|---|---|")
    for fw, data in frameworks_data.items():
        lines.append(f"| {fw} | {data['version']} |")

    lines.append("\n## Detailed Operators\n")
    lines.append("| ONNX Operator | ONNX9000 |")
    lines.append("|---|---|")
    if onnx_total > 0:
        for op in sorted(onnx_data.get("operators", [])):
            op_lower = op.lower()
            is_supported = "✅" if op_lower in onnx9000_ops else "❌"
            lines.append(f"| {op} | {is_supported} |")

    return "\n".join(lines)


def update_compliance_md(summary_md: str) -> None:
    """Inject summary table into COMPLIANCE.md.

    Args:
        summary_md: The markdown string containing the summary table.
    """
    import re

    compliance_path = os.path.abspath(os.path.join(os.getcwd(), "specs", "ONNX01_COMPLIANCE.md"))
    if not os.path.exists(compliance_path):
        return

    with open(compliance_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find where to inject: after the first header and description, before Exhaustive Parity Checklist
    # Or just replace the existing summary if it exists.
    summary_marker_start = "<!-- COVERAGE_SUMMARY_START -->"
    summary_marker_end = "<!-- COVERAGE_SUMMARY_END -->"

    injection = f"{summary_marker_start}\n{summary_md}\n{summary_marker_end}"

    if summary_marker_start in content and summary_marker_end in content:
        pattern = re.compile(f"{summary_marker_start}.*?{summary_marker_end}", re.DOTALL)
        new_content = pattern.sub(injection, content)
    else:
        # Inject after ## Description paragraph or at the end of the file
        desc_match = re.search(r"## Description\n.*?\n\n", content, re.DOTALL)
        if desc_match:
            insert_pos = desc_match.end()
            new_content = content[:insert_pos] + injection + "\n\n" + content[insert_pos:]
        else:
            new_content = content + "\n\n" + injection

    with open(compliance_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def update_coverage_cmd(args: argparse.Namespace) -> None:
    """Update the API coverage tracking JSON and markdown files.

    Args:
        args: Parsed command line arguments.
    """
    snapshots_dir = os.path.abspath(os.path.join(os.getcwd(), "snapshots"))
    os.makedirs(snapshots_dir, exist_ok=True)

    print("Collecting framework APIs via PyPI and temporary venvs...")
    frameworks_data = generate_framework_snapshots(snapshots_dir)

    print("Cloning ONNX spec and parsing markdown...")
    onnx_data = clone_and_parse_onnx_spec()
    onnx_commit = onnx_data.get("commit", "unknown")
    onnx_path = os.path.join(snapshots_dir, f"onnx-{onnx_commit}.json")
    with open(onnx_path, "w") as f:
        json.dump(onnx_data, f, indent=2)
    print(f"Saved ONNX spec to {onnx_path}")

    print("Getting ONNX9000 supported ops...")
    onnx9000_ops = get_onnx9000_ops()

    print("Generating SUPPORTED_PER_FRAMEWORK.md...")
    md_content = generate_markdown_table(frameworks_data, onnx_data, onnx9000_ops)
    md_path = os.path.abspath(os.path.join(os.getcwd(), "SUPPORTED_PER_FRAMEWORK.md"))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved coverage report to {md_path}")

    print("Updating COMPLIANCE.md...")
    summary_md = generate_summary_table(frameworks_data, onnx_data, onnx9000_ops)
    update_compliance_md(summary_md)
    print("Coverage update complete.")
