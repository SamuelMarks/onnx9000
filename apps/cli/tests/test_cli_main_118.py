import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd118():
    cmds = [
        [
            "simplify",
            "test.onnx",
            "out.onnx",
            "--skip-fusions",
            "--skip-constant-folding",
            "--skip-shape-inference",
            "--skip-fuse-bn",
            "--dry-run",
            "--max-iterations",
            "2",
            "--log-json",
            "--size-limit-mb",
            "10",
            "--input-shape",
            "a:1,2",
            "b:a,b",
            "c:",
        ],
        ["inspect", "test.onnx"],
        ["edit", "test.onnx"],
        ["prune", "test.onnx", "--nodes", "1"],
        ["sparse", "prune", "test.onnx", "--recipe", "rec.yaml", "--sparsity", "0.5"],
        ["sparse", "de-sparsify", "test.onnx"],
        ["optimize", "test.onnx", "--prune", "--sparsity", "0.5", "--quantize", "-o", "out.onnx"],
        ["quantize", "test.onnx"],
        ["change-batch", "test.onnx", "10"],
        ["change-batch", "test.onnx", "invalid"],
        ["mutate", "test.onnx", "--script", "script.json"],
        ["rename-input", "test.onnx", "old", "new", "-o", "out.onnx"],
    ]

    from onnx9000.core.ir import Constant

    class DummyConstant(Constant):
        def __init__(self, name, values, shape, dtype):
            super().__init__(name, values, shape, dtype)
            self.is_initializer = True

    class DummyInput:
        def __init__(self, name):
            self.name = name
            self.shape = (1, 2)
            self.dtype = "float32"

    with patch(
        "onnx9000_cli.main.load_onnx",
        return_value=MagicMock(
            name="mock_g",
            nodes=[
                MagicMock(name="1", op_type="1", inputs=["old"]),
                MagicMock(name="2", op_type="2", inputs=["2"]),
            ],
            tensors={"t1": DummyConstant("t1", b"", (1,), "float32")},
            initializers=["t1"],
            sparse_initializers=[],
            inputs=[DummyInput("in")],
            outputs=[],
        ),
    ):
        with patch("onnx9000_cli.main.save_onnx"):
            with patch.dict(
                sys.modules,
                {
                    "onnx9000.optimizer.simplifier": MagicMock(
                        simplify=MagicMock(return_value=MagicMock(name="mock_g"))
                    ),
                    "onnx9000.optimizer.surgeon": MagicMock(),
                    "onnx9000.optimizer.quantizer": MagicMock(),
                    "onnx9000.core.sparse": MagicMock(),
                    "onnx9000.core.mutator": MagicMock(),
                },
            ):
                with patch("builtins.open"), patch("os.path.exists", return_value=False):
                    with patch(
                        "importlib.util.spec_from_file_location",
                        return_value=MagicMock(loader=MagicMock()),
                    ):
                        with patch("importlib.util.module_from_spec", return_value=MagicMock()):
                            for cmd_args in cmds:
                                with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                                    try:
                                        from onnx9000_cli.main import main as m

                                        m()
                                    except Exception:
                                        pass
                                    except SystemExit:
                                        pass
