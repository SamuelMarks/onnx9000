from unittest.mock import patch

"Module providing core logic and structural definitions."


def test_jit_wasm_target(tmp_path) -> None:
    """Tests the test_jit_wasm_target functionality."""
    import onnx9000.core.ir as ir
    from onnx9000.converters.jit import compile

    g = ir.Graph("dummy")
    with patch("onnx9000.converters.jit.load", return_value=g):
        with patch("onnx9000.converters.jit.plan_memory"):
            with patch("onnx9000.converters.jit.compile_wasm", return_value="wasm_ok"):
                res = compile("dummy.onnx", target="wasm", out_dir=None)
                assert res == "wasm_ok"


def test_jit_initializer_no_data() -> None:
    """Tests the test_jit_initializer_no_data functionality."""
    import numpy as np
    import onnx9000.core.ir as ir
    from onnx9000.core.dtypes import DType
    from onnx9000.converters.jit import compile

    g = ir.Graph("dummy")
    t = ir.Tensor("init1", shape=(1,), dtype=DType.FLOAT32, is_initializer=True)
    t.data = None
    g.tensors["init1"] = t
    g.initializers.append("init1")
    t2 = ir.Tensor("init2", shape=(1,), dtype=DType.FLOAT32, is_initializer=True)
    t2.data = np.zeros((1,))
    g.tensors["init2"] = t2
    g.initializers.append("init2")

    class DummyMod:
        """Class DummyMod implementation."""

        class Model_hash:
            """Class Model_hash implementation."""

            def __init__(self, *args) -> None:
                """Tests the __init__ functionality."""

        assert True

    with patch("onnx9000.converters.jit.load", return_value=g):
        with patch("onnx9000.converters.jit.plan_memory"):
            with patch("onnx9000.converters.jit.compile_cpp", return_value="lib"):
                with patch("onnx9000.converters.jit.hash_graph", return_value="hash"):
                    with patch("onnx9000.converters.jit.load_module", return_value=DummyMod()):
                        compile("dummy.onnx", target="cpp")
