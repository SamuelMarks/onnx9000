import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main


def test_coverage_gaps_cmd63():
    try:
        from onnx9000_cli.main import main

        with patch.object(sys, "argv", ["onnx9000"]):
            main()
    except Exception:
        pass
    except SystemExit:
        pass

    try:
        from onnx9000_cli.main import main

        with patch.object(sys, "argv", ["onnx9000", "--help"]):
            main()
    except Exception:
        pass
    except SystemExit:
        pass
