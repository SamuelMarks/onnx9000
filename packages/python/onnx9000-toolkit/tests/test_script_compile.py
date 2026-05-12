import os
import tempfile

import pytest
from onnx9000.toolkit.script import parse_and_compile


def test_parse_and_compile():
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("""
from onnx9000.toolkit.script import script, op

@script
def test_model(x):
    return op.Relu(x)
""")
        f_path = f.name

    try:
        model = parse_and_compile(f_path)
        assert model is not None
    finally:
        os.remove(f_path)


def test_parse_and_compile_no_script():
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("""
def test_model(x):
    return x
""")
        f_path = f.name

    try:
        with pytest.raises(ValueError, match="No @script decorated function found"):
            parse_and_compile(f_path)
    finally:
        os.remove(f_path)
