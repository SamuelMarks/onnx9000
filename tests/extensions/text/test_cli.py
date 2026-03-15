"""Module providing core logic and structural definitions."""

from unittest.mock import patch
from onnx9000.extensions.text.cli import main
import sys


def test_cli_success(tmpdir):
    """Provides semantic functionality and verification."""
    json_path = tmpdir.join("tokenizer.json")
    bin_path = tmpdir.join("tokenizer.bin")
    with open(json_path, "w") as f:
        f.write('{"model": {"vocab": {"a": 1}, "merges": ["a b"]}}')
    with patch.object(sys, "argv", ["cli", str(json_path), str(bin_path)]):
        main()
    assert bin_path.exists()


def test_cli_failure(tmpdir, capsys):
    """Provides semantic functionality and verification."""
    with patch.object(sys, "argv", ["cli", "nonexistent.json", "out.bin"]):
        try:
            main()
        except SystemExit:
            pass
        out, err = capsys.readouterr()
        assert "Failed to export" in err
