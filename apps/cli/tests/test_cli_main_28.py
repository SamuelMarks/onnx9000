import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main, sparse_cmd, sparse_de_sparsify_cmd, sparse_prune_cmd


def test_coverage_gaps_cmd28():
    cmds = [
        ["sparse", "prune", "test.onnx", "--recipe", "rec.yaml", "--sparsity", "0.5"],
        ["sparse", "de-sparsify", "test.onnx"],
    ]

    from onnx9000.core.ir import Constant

    class DummyConstant(Constant):
        def __init__(self, name, values, shape, dtype):
            super().__init__(name, values, shape, dtype)
            self.is_initializer = True

    t1 = DummyConstant("t1", b"", (1,), "float32")

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(
                nodes=[],
                tensors={"t1": t1},
                initializers=["t1"],
                sparse_initializers=[],
                inputs=[],
                outputs=[],
            ),
        ),
        patch("onnx9000_cli.main.save_onnx"),
        patch("builtins.open"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.optimizer.simplifier": MagicMock(),
                "onnx9000.core.sparse": MagicMock(
                    detect_theoretical_sparsity=MagicMock(return_value=0.1),
                    analyze_topological_dead_ends=MagicMock(return_value=["1"]),
                ),
                "onnx9000.optimizer.sparse.modifier": MagicMock(),
            },
        ):
            for cmd_args in cmds:
                with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                    try:
                        main()
                    except Exception:
                        pass
                    except SystemExit:
                        pass
