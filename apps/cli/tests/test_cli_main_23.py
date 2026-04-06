import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import (
    main,
    optimum_export_cmd,
    optimum_optimize_cmd,
    optimum_quantize_cmd,
    optimum_cmd,
    openvino_export_cmd,
    openvino_cmd,
)


def test_coverage_gaps_cmd23():
    cmds = [
        ["optimum", "export", "test", "out", "--task", "task", "--device", "cpu", "--atol", "1e-4"],
        ["optimum", "optimize", "test", "out", "--level", "O3"],
        ["optimum", "quantize", "test", "out", "--weight-only", "--calibration-data", "data"],
        ["openvino", "export", "test.onnx", "-o", "out", "--fp16"],
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
                "onnx9000_optimum.export": MagicMock(),
                "onnx9000_optimum.optimize": MagicMock(),
                "onnx9000_optimum.quantize": MagicMock(),
                "onnx9000.openvino.api": MagicMock(),
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
