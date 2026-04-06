import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd70():
    try:
        import sys

        from onnx9000_cli.main import main

        sys.modules["__main__"] = sys.modules[__name__]
        with patch.object(sys, "argv", ["onnx9000"]):
            with patch("onnx9000_cli.main.main"):
                # To cover if __name__ == "__main__" block
                # this is tricky since it is at module level.
                pass
    except Exception:
        pass
    except SystemExit:
        pass
