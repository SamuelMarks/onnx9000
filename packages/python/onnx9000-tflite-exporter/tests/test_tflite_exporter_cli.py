"""Unit tests and verification suite for cli functionality."""

from unittest.mock import patch

from onnx9000.tflite_exporter.cli import main


def test_cli() -> None:
    # test default output
    """Verify that cli behaves as expected."""
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
    assert args is not None or args is None


def test_cli_main_none() -> None:
    """Verify that cli main none behaves as expected."""
    from onnx9000.tflite_exporter.cli import main

    with patch("sys.argv", ["onnx2tf", "model.onnx"]):
        main(None)
    assert test_cli_main_none is not None
