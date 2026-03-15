"""Module providing core logic and structural definitions."""

import pytest
import json
from onnx9000.extensions.text.exporter import export_tokenizer_binary
from onnx9000.extensions.text.loader import BinaryLoader


def test_binary_export_load(tmpdir):
    """Provides semantic functionality and verification."""
    json_path = tmpdir.join("tokenizer.json")
    bin_path = tmpdir.join("tokenizer.bin")
    data = {
        "model": {"vocab": {"a": 1, "b": 2, "ab": 3}, "merges": ["a b"]},
        "added_tokens": [{"id": 0, "content": "[UNK]"}],
    }
    with open(json_path, "w") as f:
        json.dump(data, f)
    export_tokenizer_binary(str(json_path), str(bin_path))
    tokenizer = BinaryLoader.load_bpe_binary(str(bin_path))
    assert tokenizer.vocab["a"] == 1
    assert tokenizer.vocab["[UNK]"] == 0
    assert tokenizer.merges["a", "b"] == 0


def test_binary_invalid_magic(tmpdir):
    """Provides semantic functionality and verification."""
    bin_path = tmpdir.join("tokenizer.bin")
    with open(bin_path, "wb") as f:
        f.write(b"BADMAGIC12")
    with pytest.raises(ValueError):
        BinaryLoader.load_bpe_binary(str(bin_path))
