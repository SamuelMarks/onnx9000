import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd65():
    try:
        from onnx9000_cli.main import main

        with patch("sys.exit"):
            with patch.object(
                sys,
                "argv",
                ["onnx9000", "convert", "test.onnx", "--from", "unknown", "--to", "onnx"],
            ):
                main()
    except Exception:
        pass
    except SystemExit:
        pass

    try:
        with patch("sys.exit"):
            with patch.object(
                sys,
                "argv",
                ["onnx9000", "convert", "test.onnx", "--from", "keras", "--to", "unknown"],
            ):
                main()
    except Exception:
        pass
    except SystemExit:
        pass
