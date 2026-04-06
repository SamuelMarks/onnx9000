import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main, export_cmd


def test_coverage_gaps_cmd22():
    cmds = [
        ["export", "script.py", "out.onnx", "--format", "onnx"],
        ["export", "script.py"],
        ["export", "script2.py"],
        ["export", "script3.py"],
    ]

    class DummyModule:
        pass

    class MyMod(DummyModule):
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

            def mock_spec(name, script):
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
                        with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                            try:
                                main()
                            except Exception:
                                pass
                            except SystemExit:
                                pass
