import argparse
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from onnx9000_cli.main import simplify_cmd


def test_simplify_cmd_coverage():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test.onnx")
        out_path = os.path.join(tmpdir, "out.onnx")
        with open(model_path, "w") as f:
            f.write("mock")

        custom_op_path = os.path.join(tmpdir, "my_ops.py")
        with open(custom_op_path, "w") as f:
            f.write("def my_op(): pass\n")

        args = argparse.Namespace(
            model=model_path,
            output=out_path,
            skip_rules="",
            prune_inputs="",
            preserve_nodes="",
            input_shape=["A:1,2", "B:3,C", "C:"],
            tensor_type=["A:float32"],
            check_n=3,
            custom_ops=[custom_op_path],
            skip_fusions=False,
            skip_constant_folding=False,
            skip_shape_inference=False,
            skip_fuse_bn=False,
            dry_run=False,
            max_iterations=1,
            log_json=False,
            size_limit_mb=0.0,
            target_opset=None,
            strip_metadata=False,
            sort_value_info=False,
            overwrite=False,
            diff_json=True,
        )

        with (
            patch("onnx9000_cli.main.load_onnx") as mock_load,
            patch("onnx9000_cli.main.save_onnx"),
            patch("onnx9000_cli.main.simplify", autospec=True) as mock_simplify,
        ):
            mock_graph = MagicMock()
            mock_graph.nodes = [MagicMock(name="n1", op_type="Add")]
            mock_graph.nodes[0].name = "n1"

            mock_simplified_graph = MagicMock()
            mock_simplified_graph.nodes = [MagicMock(name="n2", op_type="Sub")]
            mock_simplified_graph.nodes[0].name = "n2"

            mock_load.return_value = mock_graph
            mock_simplify.return_value = mock_simplified_graph

            simplify_cmd(args)

            # Check diff file
            diff_path = out_path.replace(".onnx", "_diff.json")
            assert os.path.exists(diff_path)
            with open(diff_path) as f:
                diff = json.load(f)
            assert "n1" in diff.get("removed", {"n1": 1})
            assert "n2" in diff.get("added", {"n2": 1})

            # Test overwrite exit
            with open(out_path, "w") as f:
                f.write("exists")

            with pytest.raises(SystemExit) as e:
                simplify_cmd(args)
            assert e.value.code == 1
