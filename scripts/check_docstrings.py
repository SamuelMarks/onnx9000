"""Enforce 100.0% Python docstring coverage across modules, classes, and functions.

Validates that every Python module, class, method, and function (including private
helpers and unit tests) has a non-empty, non-boilerplate docstring.
"""

from __future__ import annotations

import ast
import glob
import re
import sys

BOILERPLATE_RE: re.Pattern[str] = re.compile(
    r"^(Auto-generated doc\.|Provide functionality for this module\.|"
    r"Implement the .* method(?: or operation)?\.|Initialize(?:\.| the instance\.)?|"
    r"Init\.|Call\.|Do the function\.|Docstring for D\d+\.|Magic method\.|"
    r"Module containing .* definitions\.|"
    r"Module providing core logic and structural definitions\.)$",
    re.IGNORECASE,
)

EXCLUDED_SEGMENTS: tuple[str, ...] = (
    ".venv/",
    "node_modules/",
    "pycdc/",
    "__pycache__/",
    ".git/",
    "build/",
    "dist/",
)


def check_file(filepath: str) -> list[str]:
    """Validate that a single Python file has complete, non-boilerplate docstrings.

    Args:
        filepath: Relative or absolute path to the Python file to inspect.

    Returns:
        A list of human-readable diagnostic messages for any missing or boilerplate docstrings.
    """
    with open(filepath, encoding="utf-8") as handle:
        source = handle.read()
    if not source.strip():
        return [f"{filepath}:1: Empty file missing module docstring"]

    tree = ast.parse(source, filename=filepath)
    errors: list[str] = []

    mod_doc = ast.get_docstring(tree)
    if not mod_doc:
        errors.append(f"{filepath}:1: Missing module docstring")
    elif BOILERPLATE_RE.match(mod_doc.strip().splitlines()[0]):
        errors.append(
            f"{filepath}:1: Boilerplate module docstring: {mod_doc.strip().splitlines()[0]!r}"
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            if not doc:
                errors.append(
                    f"{filepath}:{node.lineno}: Missing {kind} docstring for '{node.name}'"
                )
            elif BOILERPLATE_RE.match(doc.strip().splitlines()[0]):
                first_line = doc.strip().splitlines()[0]
                errors.append(
                    f"{filepath}:{node.lineno}: Boilerplate {kind} docstring for '{node.name}': {first_line!r}"
                )
    return errors


def main() -> int:
    """Scan all repository Python files and return non-zero if docstring coverage is below 100%.

    Returns:
        Exit status code (0 if 100.0% compliant, 1 otherwise).
    """
    py_files = [
        f
        for f in sorted(glob.glob("**/*.py", recursive=True))
        if not any(seg in f for seg in EXCLUDED_SEGMENTS) and not f.endswith("onnx_pb2.py")
    ]
    all_errors: list[str] = []
    for path in py_files:
        all_errors.extend(check_file(path))

    if all_errors:
        for err in all_errors[:50]:
            print(err, file=sys.stderr)
        print(
            f"Found {len(all_errors)} docstring coverage errors across {len(py_files)} files.",
            file=sys.stderr,
        )
        return 1

    print(f"100.0% docstring coverage verified across {len(py_files)} Python files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
