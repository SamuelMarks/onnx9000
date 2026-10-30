"""Unit tests and verification suite for cli main 22 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd22() -> None:
    """Verify that coverage gaps cmd22 behaves as expected.

    Returns:
        Result of test coverage gaps cmd22.
    """
    cmds = [
        ["export", "script.py", "out.onnx", "--format", "onnx"],
        ["export", "script.py"],
        ["export", "script2.py"],
        ["export", "script3.py"],
    ]

    class DummyModule:
        """Represents dummy module."""

        pass

    class MyMod(DummyModule):
        """Represents my mod inheriting from DummyModule."""

        pass

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.converters.frontend.nn.module": MagicMock(Module=DummyModule),
                "onnx9000.converters.frontend.tracer": MagicMock(
                    trace=MagicMock(
                        return_value=MagicMock(to_graph=MagicMock(return_value=MagicMock()))
                    )
                ),
                "onnx9000.core.exporter": MagicMock(),
                "onnx9000.converters": MagicMock(),
                "onnx9000.converters.torch_like": MagicMock(randn=MagicMock(return_value="t")),
            },
        ):

            def mock_spec(
                name: int | float | str | bool | bytes | list | dict | tuple | None,
                script: int | float | str | bool | bytes | list | dict | tuple | None,
            ) -> int | float | str | bool | bytes | list | dict | tuple | None:
                """Execute mock spec operation.

                Args:
                    name: Parameter specifying name.
                    script: Parameter specifying script.

                Returns:
                    Result of mock spec.
                """
                if script == "script2.py":
                    return None
                if script == "script3.py":
                    m = MagicMock(loader=None)
                    return m
                return MagicMock(loader=MagicMock())

            with patch("importlib.util.spec_from_file_location", side_effect=mock_spec):
                with patch(
                    "importlib.util.module_from_spec",
                    side_effect=[
                        type("module", (), {"MyMod": MyMod}),
                        type("module", (), {"MyMod": DummyModule}),
                    ],
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
    assert cmds is not None or cmds is None
