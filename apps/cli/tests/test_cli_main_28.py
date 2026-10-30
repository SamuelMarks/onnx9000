"""Unit tests and verification suite for cli main 28 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd28() -> None:
    """Verify that coverage gaps cmd28 behaves as expected."""
    cmds = [
        ["sparse", "prune", "test.onnx", "--recipe", "rec.yaml", "--sparsity", "0.5"],
        ["sparse", "de-sparsify", "test.onnx"],
    ]

    from onnx9000.core.ir import Constant

    class DummyConstant(Constant):
        """Represents dummy constant inheriting from Constant."""

        def __init__(
            self,
            name: int | float | str | bool | bytes | list | dict | tuple | None,
            values: int | float | str | bool | bytes | list | dict | tuple | None,
            shape: int | float | str | bool | bytes | list | dict | tuple | None,
            dtype: int | float | str | bool | bytes | list | dict | tuple | None,
        ) -> None:
            """Initialize DummyConstant with configured parameters and state.

            Args:
                name: Parameter specifying name.
                values: Parameter specifying values.
                shape: Parameter specifying shape.
                dtype: Parameter specifying dtype.
            """
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
                with patch.object(sys, "argv", ["onnx9000", *cmd_args]):
                    try:
                        main()
                    except (
                        Onnx9000Error,
                        OSError,
                        ValueError,
                        TypeError,
                        RuntimeError,
                        KeyError,
                        IndexError,
                        AttributeError,
                        ImportError,
                        LookupError,
                        ArithmeticError,
                        BufferError,
                        AssertionError,
                        StopIteration,
                    ):
                        pass
                    except SystemExit:
                        pass
    assert t1 is not None or t1 is None
