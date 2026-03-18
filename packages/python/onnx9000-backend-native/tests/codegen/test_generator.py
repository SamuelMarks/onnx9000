"""Tests the generator module functionality."""

from unittest.mock import patch

import onnx9000.backends.codegen.generator as cg
import onnx9000.backends.codegen.op_generator as cog
import onnx9000.backends.codegen.utils as cu
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor


def test_sanitize_name() -> None:
    """Tests the sanitize name functionality."""
    assert cu.sanitize_name("1name") == "var_1name"
    assert cu.sanitize_name("my.name-!") == "my_name__"
    assert cu.sanitize_name("good_name") == "good_name"


def test_get_omp_pragma() -> None:
    """Tests the get omp pragma functionality."""
    with patch("onnx9000.core.config.ONNX9000_USE_CUDA", True):
        assert cu.get_omp_pragma("N") == ""

    with patch("onnx9000.core.config.ONNX9000_USE_CUDA", False):
        pragma = cu.get_omp_pragma("N")
        assert "__EMSCRIPTEN__" in pragma
        assert "omp parallel for" in pragma


class MockOpGen(cog.OpGenerator):
    """Represents the Mock Op Gen class."""

    def generate(self, node, generator_context) -> str:
        """Generate mock code for an operation."""
        return f"// Mock OP for {node.op_type}"


def test_generator() -> None:
    """Tests the generator functionality."""
    g = Graph("test")
    g.inputs.append("in1")
    g.inputs.append("in1")  # test seen
    t_in = Tensor("in1", (2, 2), DType.FLOAT32)
    t_in.buffer_id = 0
    g.tensors["in1"] = t_in

    g.outputs.append("out1")
    g.outputs.append("out2")
    t_out1 = Tensor("out1", (2, 2), DType.FLOAT32)
    t_out1.buffer_id = 1
    g.tensors["out1"] = t_out1

    t_out2 = Tensor("out2", (2, 2), DType.FLOAT32)
    t_out2.buffer_id = None  # Test branch
    g.tensors["out2"] = t_out2

    g.initializers.append("init1")
    t_init = Tensor("init1", (2,), DType.FLOAT32)
    t_init.buffer_id = 2
    g.tensors["init1"] = t_init

    n = Node("Add", ["in1", "init1"], ["out1"])
    g.add_node(n)

    with patch("onnx9000.core.registry.global_registry.get_op") as mock_get_gen:
        mock_get_gen.return_value = MockOpGen().generate

        with patch("onnx9000.core.config.ONNX9000_USE_CUDA", False):
            gen = cg.Generator(g)
            code = gen.generate()
            assert "class GeneratedModel" in code
            assert "std::vector<uint8_t> _global_arena;" in code
            assert "// Mock OP for Add" in code
            assert gen.get_tensor_name("111") == "var_111"

        with patch("onnx9000.core.config.ONNX9000_USE_CUDA", True):
            gen2 = cg.Generator(g)
            code2 = gen2.generate()
            assert "onnx9000::CudaBuffer _global_arena;" in code2

        # test empty outputs returning void or tuple
        g.outputs.clear()
        gen3 = cg.Generator(g)
        code3 = gen3.generate()
        assert "void forward_py(" in code3


import os
import shutil
import tempfile

import numpy as np


def test_generator_constants() -> None:
    """Tests the generator constants functionality."""
    g = Graph("test_graph")
    g.tensors["init_tensor"] = Tensor(
        "init_tensor",
        (2,),
        DType.FLOAT32,
        is_initializer=True,
        data=b"\x00\x00\x00\x00\x00\x00\x00\x00",
    )
    g.tensors["init_values"] = Tensor("init_values", (2,), DType.FLOAT32, is_initializer=True)
    g.tensors["init_values"].values = [1.0, 2.0]
    g.tensors["init_int64"] = Tensor("init_int64", (1,), DType.INT64, is_initializer=True)
    g.tensors["init_int64"].values = [42]
    g.tensors["init_int32"] = Tensor("init_int32", (1,), DType.INT32, is_initializer=True)
    g.tensors["init_int32"].values = [42]
    g.tensors["init_bool"] = Tensor("init_bool", (1,), DType.BOOL, is_initializer=True)
    g.tensors["init_bool"].values = [True]

    g.initializers.append("init_tensor")
    g.initializers.append("init_values")
    g.initializers.append("init_int64")
    g.initializers.append("init_int32")
    g.initializers.append("init_bool")

    gen = cg.Generator(g)

    # Test embed_constants
    code_embed = gen.generate(embed_constants=True)
    assert "init_tensor_data[]" in code_embed
    assert "init_values_data[]" in code_embed

    # Test mmap_constants
    with tempfile.TemporaryDirectory() as temp_dir:
        mmap_path = os.path.join(temp_dir, "weights.bin")
        code_mmap = gen.generate(mmap_constants=True, mmap_file=mmap_path)
        assert os.path.exists(mmap_path)
        assert "_mmap_ptr" in code_mmap


def test_generator_write_to_directory() -> None:
    """Tests the generator write to directory functionality."""
    g = Graph("test_graph")
    gen = cg.Generator(g)

    with tempfile.TemporaryDirectory() as temp_dir:
        cpp_path = gen.write_to_directory(temp_dir, run_clang_format=False)
        assert os.path.exists(cpp_path)
        assert os.path.exists(os.path.join(temp_dir, "CMakeLists.txt"))
        assert os.path.exists(os.path.join(temp_dir, "main.cpp"))

        # Test clang-format invocation
        with patch("subprocess.run") as mock_run:
            gen.write_to_directory(temp_dir, run_clang_format=True)
            mock_run.assert_called()

        # Test clang-format not found
        with patch("subprocess.run", side_effect=FileNotFoundError):
            gen.write_to_directory(temp_dir, run_clang_format=True)  # Should not raise
