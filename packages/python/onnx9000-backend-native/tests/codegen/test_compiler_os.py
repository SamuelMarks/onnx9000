import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest
from onnx9000.backends.codegen.compiler import compile_cpp, compile_static_lib


def test_compile_cpp_os_branches():
    with patch("sys.platform", "win32"):
        with patch("subprocess.run"):
            # use_pybind=False, win32 -> .dll
            so_path = compile_cpp("int main(){}", use_pybind=False)
            assert so_path.endswith(".dll")

            # use_pybind=True, win32 -> .pyd
            so_path_py = compile_cpp("int main(){}", use_pybind=True)
            assert so_path_py.endswith(".pyd")

    with patch("sys.platform", "linux"):
        with patch("subprocess.run"):
            # use_pybind=False, linux -> .so
            so_path = compile_cpp("int main(){}", use_pybind=False)
            assert so_path.endswith(".so")

            # use_pybind=True, linux -> .so
            so_path_py = compile_cpp("int main(){}", use_pybind=True)
            assert so_path_py.endswith(".so")


def test_compile_static_lib():
    with patch("subprocess.run"):
        # We need an ar compiler mock
        with patch("sys.platform", "win32"):
            obj_path = compile_static_lib("int main(){}")
            assert obj_path.endswith(".lib")

        with patch("sys.platform", "darwin"):
            obj_path = compile_static_lib("int main(){}")
            assert obj_path.endswith(".a")

    # Test compiler failure
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd", stderr="err")):
        with pytest.raises(RuntimeError, match="Object compilation failed"):
            compile_static_lib("invalid")

    # Test archiving failure
    with patch(
        "subprocess.run", side_effect=[None, subprocess.CalledProcessError(1, "cmd", stderr="err")]
    ):
        with pytest.raises(RuntimeError, match="Archiving failed"):
            compile_static_lib("invalid")
