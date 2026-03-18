import pytest
from unittest.mock import patch
import os
import warnings
from onnx9000.backends.codegen.compiler import compile_wasm, compile_static_lib


def test_wasm_size_warning():
    with patch("subprocess.run"), patch("os.path.exists", return_value=True):
        with patch("os.path.getsize", return_value=5 * 1024 * 1024):  # 5MB
            with pytest.warns(UserWarning, match="exceeds standard 4MB"):
                compile_wasm("int main(){}")


def test_compile_static_lib_cleanup():
    # If the object path exists, it should be removed
    with patch("subprocess.run"):
        with patch("os.path.exists", side_effect=lambda p: True):
            with patch("os.remove") as mock_rm:
                compile_static_lib("int main(){}")
                mock_rm.assert_called()
