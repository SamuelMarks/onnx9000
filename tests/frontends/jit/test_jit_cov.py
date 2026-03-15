"""Module providing core logic and structural definitions."""


def test_jit_wasm_target(tmp_path):
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.jit import compile
    from unittest.mock import patch
    import onnx9000.core.ir as ir

    g = ir.Graph("dummy")
    with patch("onnx9000.frontends.jit.load", return_value=g):
        with patch("onnx9000.frontends.jit.plan_memory"):
            with patch("onnx9000.frontends.jit.compile_wasm", return_value="wasm_ok"):
                res = compile("dummy.onnx", target="wasm", out_dir=None)
                assert res == "wasm_ok"


def test_jit_initializer_no_data():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.jit import compile
    from unittest.mock import patch
    import onnx9000.core.ir as ir
    from onnx9000.core.dtypes import DType
    import numpy as np

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
        """Represents the DummyMod class."""

        class Model_hash:
            """Represents the Model hash class."""

            def __init__(self, *args):
                """Provides   init   functionality and verification."""
                pass

    with patch("onnx9000.frontends.jit.load", return_value=g):
        with patch("onnx9000.frontends.jit.plan_memory"):
            with patch("onnx9000.frontends.jit.compile_cpp", return_value="lib"):
                with patch("onnx9000.frontends.jit.hash_graph", return_value="hash"):
                    with patch(
                        "onnx9000.frontends.jit.load_module", return_value=DummyMod()
                    ):
                        compile("dummy.onnx", target="cpp")
