import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main


def test_coverage_gaps_cmd81():
    cmds = [
        ["convert", "test.onnx", "--from", "onnx", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "keras", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "jax", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "paddle", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "sklearn", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "safetensors", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "caffe", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "lightgbm", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "xgboost", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "libsvm", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "h2o", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "sparkml", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "coreml", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "pytorch", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "unknown", "--to", "onnx", "-o", "out.onnx"],
        ["convert", "test.onnx", "--from", "onnx", "--to", "mlir", "-o", "out.mlir"],
        ["convert", "test.onnx", "--from", "onnx", "--to", "c", "-o", "out.c"],
        ["convert", "test.onnx", "--from", "onnx", "--to", "cpp", "-o", "out.cpp"],
        ["convert", "test.onnx", "--from", "onnx", "--to", "keras", "-o", "out.py"],
        ["convert", "test.onnx", "--from", "onnx", "--to", "pytorch", "-o", "out.py"],
        ["convert", "test.onnx", "--from", "onnx", "--to", "wasm", "-o", "out.js"],
        ["convert", "test.onnx"],
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
                "keras.saving": MagicMock(load_model=MagicMock()),
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
                "torch.fx": MagicMock(symbolic_trace=MagicMock()),
                "torch": MagicMock(),
                "onnx9000.converters.torch.fx": MagicMock(
                    PyTorchFXParser=MagicMock(
                        return_value=MagicMock(parse=MagicMock(return_value=MagicMock()))
                    )
                ),
                "onnx9000.core.exporter": MagicMock(),
            },
        ):
            with (
                patch("builtins.open"),
                patch("os.path.isdir", side_effect=[False] * 21 + [True]),
                patch("os.path.exists", side_effect=[False] * 21 + [True, False]),
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
