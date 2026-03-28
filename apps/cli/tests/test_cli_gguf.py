"""Tests for the ONNX and GGUF conversion commands within the CLI.

This module validates the behavior of the `onnx2gguf_cmd` and `gguf2onnx_cmd`
functions, ensuring that dry runs, large model warnings, and standard conversion
executions are properly initiated with the correct file operations and compiler logic.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest
from onnx9000_cli.main import gguf2onnx_cmd, onnx2gguf_cmd


def test_onnx2gguf_dry_run(capsys):
    """Test the ONNX to GGUF conversion command in dry-run mode.

    Verifies that when the `dry_run` flag is provided, the command prints the
    dry-run status to stdout and avoids executing actual heavy conversion tasks.
    """
    args = argparse.Namespace(
        model="dummy.onnx",
        output=None,
        tokenizer=None,
        outtype=None,
        architecture=None,
        dry_run=True,
        force=False,
    )
    onnx2gguf_cmd(args)
    out, _ = capsys.readouterr()
    assert "Dry run" in out


def test_onnx2gguf_massive(monkeypatch, capsys):
    """Ensure that massive models trigger a warning during ONNX to GGUF conversion.

    By mocking the file size to be exceedingly large, this test confirms that
    the command correctly detects the file magnitude and outputs the appropriate
    "Massive model detected" warning to stdout.
    """
    monkeypatch.setattr("os.path.exists", lambda x: True)
    monkeypatch.setattr("os.path.getsize", lambda x: 80_000_000_000)
    args = argparse.Namespace(
        model="dummy.onnx",
        output=None,
        tokenizer=None,
        outtype=None,
        architecture=None,
        dry_run=False,
        force=False,
    )
    onnx2gguf_cmd(args)
    out, _ = capsys.readouterr()
    assert "Massive model detected" in out


def test_onnx2gguf(monkeypatch):
    """Verify standard execution of the ONNX to GGUF conversion process.

    This test sets up a mock environment to simulate a normal-sized ONNX file,
    mocking file parsers and compilers, and asserts that the internal GGUF
    compiler routine is successfully invoked exactly once.
    """
    monkeypatch.setattr("os.path.getsize", lambda x: 100)

    mock_graph = MagicMock()
    monkeypatch.setattr("onnx9000_cli.main.load_onnx", lambda x: mock_graph)

    import io

    monkeypatch.setattr("builtins.open", lambda *a, **kw: io.StringIO('{"dummy": 1}'))
    called = []
    monkeypatch.setattr(
        "onnx9000.onnx2gguf.compiler.compile_gguf", lambda g, f, kv_overrides: called.append(True)
    )

    args = argparse.Namespace(
        model="dummy.onnx",
        output="dummy.gguf",
        tokenizer="tok.json",
        outtype="f16",
        architecture="llama",
        dry_run=False,
        force=False,
    )
    onnx2gguf_cmd(args)
    assert len(called) == 1


def test_gguf2onnx(monkeypatch):
    """Validate the GGUF to ONNX reverse conversion execution.

    By replacing the GGUF reader and ONNX reconstructor with dummy objects,
    this checks that the parsed output correctly routes to the final serialization
    function, saving the resulting graph structure to the target path.
    """
    monkeypatch.setattr("onnx9000.onnx2gguf.reader.GGUFReader", lambda f: "reader")

    mock_graph = MagicMock()
    mock_graph.doc_string = ""
    mock_graph.metadata_props = {}
    mock_graph.opset_imports = {}
    mock_graph.tensors = {}
    mock_graph.initializers = []
    mock_graph.sparse_initializers = []
    mock_graph.nodes = []
    mock_graph.inputs = []
    mock_graph.outputs = []
    mock_graph.value_info = []

    monkeypatch.setattr("onnx9000.onnx2gguf.reverse.reconstruct_onnx", lambda r: mock_graph)

    called = []
    monkeypatch.setattr("onnx9000_cli.main.save_onnx", lambda g, f: called.append((g, f)))

    import io

    monkeypatch.setattr("builtins.open", lambda *a, **kw: io.StringIO('{"dummy": 1}'))
    args = argparse.Namespace(model="dummy.gguf", output="dummy.onnx")
    gguf2onnx_cmd(args)
    assert len(called) == 1
    assert called[0][0] == mock_graph
    assert called[0][1] == "dummy.onnx"
