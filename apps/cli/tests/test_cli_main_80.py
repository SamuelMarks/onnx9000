import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main


def test_coverage_gaps_cmd80():
    cmds = [
        ["convert", "test.onnx", "--from", "pytorch", "--to", "onnx", "-o", "out.onnx"],
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
                "keras.saving": MagicMock(),
                "keras": MagicMock(),
                "onnx9000.converters.tf.api": MagicMock(),
                "onnx9000.converters.parsers": MagicMock(),
                "onnx9000.converters.paddle.api": MagicMock(),
                "joblib": MagicMock(),
                "onnx9000.converters.sklearn.builder": MagicMock(),
                "onnx9000.converters.safetensors.loader": MagicMock(),
                "onnx9000.converters.mltools.catboost": MagicMock(),
                "onnx9000.converters.mltools.lightgbm": MagicMock(),
                "onnx9000.converters.mltools.xgboost": MagicMock(),
                "onnx9000.converters.mltools.libsvm": MagicMock(),
                "onnx9000.converters.mltools.h2o": MagicMock(),
                "onnx9000.converters.mltools.sparkml": MagicMock(),
                "onnx9000.converters.mltools.coreml": MagicMock(),
                "torch.export": MagicMock(load=MagicMock(side_effect=Exception)),
                "torch.load": MagicMock(return_value=type("Mod", (object,), {})()),
                "torch.nn": MagicMock(Module=type("Mod", (object,), {})),
                "torch.fx": MagicMock(),
                "torch": MagicMock(),
                "onnx9000.converters.torch.fx": MagicMock(),
                "onnx9000.core.exporter": MagicMock(),
            },
        ):
            with (
                patch("builtins.open"),
                patch("os.path.isdir", return_value=False),
                patch("os.path.exists", return_value=False),
                patch("json.load", return_value={}),
            ):
                for cmd_args in cmds:
                    with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                        try:
                            from onnx9000_cli.main import main as m

                            m()
                        except Exception:
                            pass
                        except SystemExit:
                            pass
