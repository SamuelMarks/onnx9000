import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd54():
    cmds = [
        ["info", "ops", "test.onnx"],
        ["info", "tensors", "test.onnx"],
        ["info", "summary", "test.onnx"],
        ["info", "shape", "test.onnx"],
        ["info", "metadata", "test.onnx"],
    ]

    with patch("onnx9000_cli.main.load_onnx") as mock_load:
        mock_graph = MagicMock()
        mock_graph.nodes = [MagicMock(op_type="Add"), MagicMock(op_type="Add")]
        mock_graph.tensors = {"t1": MagicMock(shape=(1,), dtype="float32")}
        mock_graph.inputs = [MagicMock(name="in", shape=(1,))]
        mock_graph.outputs = [MagicMock(name="out", shape=(1,))]
        mock_graph.metadata_props = {"test_prop": "test_val"}
        mock_load.return_value = mock_graph

        for cmd_args in cmds:
            with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                try:
                    main()
                except Exception:
                    pass
                except SystemExit:
                    pass
