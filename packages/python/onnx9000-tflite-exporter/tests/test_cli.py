from unittest.mock import MagicMock, patch

import pytest
from onnx9000.tflite_exporter.cli import main


def test_cli():
    # test default output
    with patch("sys.argv", ["onnx2tf", "model.onnx"]):
        main()

    # test arguments
    args = [
        "model.onnx",
        "-o",
        "out.tflite",
        "--keep-nchw",
        "--int8",
        "-b",
        "4",
        "--disable-optimization",
        "--external-weights",
        "weights.bin",
        "--progress",
        "--micro",
    ]
    main(args)

    args = ["model.onnx", "--fp16"]
    main(args)


def test_cli_main_none():
    from onnx9000.tflite_exporter.cli import main

    with patch("sys.argv", ["onnx2tf", "model.onnx"]):
        main(None)
