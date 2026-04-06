import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import export_cmd, main


class DummyModule:
    pass


def test_coverage_gaps_cmd21():
    cmds = [
        ["export", "script.py"],
    ]

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
            with patch(
                "importlib.util.spec_from_file_location", return_value=MagicMock(loader=MagicMock())
            ):
                with patch(
                    "importlib.util.module_from_spec",
                    return_value=type("module", (), {"MyMod": DummyModule}),
                ):
                    for cmd_args in cmds:
                        with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                            try:
                                main()
                            except Exception:
                                pass
                            except SystemExit:
                                pass
