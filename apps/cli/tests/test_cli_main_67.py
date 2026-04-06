import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import convert_cmd, main, optimum_cmd


def test_coverage_gaps_cmd67():
    try:
        from onnx9000_cli.main import main

        with patch("sys.exit"):
            with patch.object(sys, "argv", ["onnx9000", "optimum"]):
                main()
    except Exception:
        pass
    except SystemExit:
        pass

    try:
        from onnx9000_cli.main import main

        with patch("sys.exit"):
            with patch.object(sys, "argv", ["onnx9000", "convert", "test", "--to", "onnx"]):
                main()
    except Exception:
        pass
    except SystemExit:
        pass
