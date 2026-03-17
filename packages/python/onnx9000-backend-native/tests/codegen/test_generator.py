from unittest.mock import patch

import onnx9000.backends.codegen.generator as cg
import onnx9000.backends.codegen.op_generator as cog
import onnx9000.backends.codegen.utils as cu
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor


def test_sanitize_name() -> None:
    assert cu.sanitize_name("1name") == "var_1name"
    assert cu.sanitize_name("my.name-!") == "my_name__"
    assert cu.sanitize_name("good_name") == "good_name"


def test_get_omp_pragma() -> None:
    with patch("onnx9000.core.config.ONNX9000_USE_CUDA", True):
        assert cu.get_omp_pragma("N") == ""

    with patch("onnx9000.core.config.ONNX9000_USE_CUDA", False):
        pragma = cu.get_omp_pragma("N")
        assert "__EMSCRIPTEN__" in pragma
        assert "omp parallel for" in pragma


class MockOpGen(cog.OpGenerator):
    def generate(self, node, generator_context) -> str:
        return f"// Mock OP for {node.op_type}"


def test_generator() -> None:
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
            assert "std::vector<std::vector<uint8_t>> _arena;" in code
            assert "// Mock OP for Add" in code
            assert gen.get_tensor_name("111") == "var_111"

        with patch("onnx9000.core.config.ONNX9000_USE_CUDA", True):
            gen2 = cg.Generator(g)
            code2 = gen2.generate()
            assert "std::vector<onnx9000::CudaBuffer> _arena;" in code2

        # test empty outputs returning void or tuple
        g.outputs.clear()
        gen3 = cg.Generator(g)
        code3 = gen3.generate()
        assert "void forward_py(" in code3


def test_generator_one_output() -> None:
    g = Graph("test")
    g.outputs.append("out1")
    t_out1 = Tensor("out1", (2, 2), DType.FLOAT32)
    t_out1.buffer_id = 1
    g.tensors["out1"] = t_out1

    with patch("onnx9000.core.config.ONNX9000_USE_CUDA", False):
        gen = cg.Generator(g)
        code = gen.generate()
        assert "return out1_out_arr_0" in code
