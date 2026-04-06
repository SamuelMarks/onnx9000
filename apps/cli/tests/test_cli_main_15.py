import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import (
    main,
    simplify_cmd,
    optimize_cmd,
    quantize_cmd,
    inspect_cmd,
    edit_cmd,
    prune_cmd,
    sparse_prune_cmd,
    sparse_de_sparsify_cmd,
    sparse_cmd,
    change_batch_cmd,
    mutate_cmd,
    rename_input_cmd,
)


class DummyNode:
    def __init__(self, name, op_type, inputs):
        self.name = name
        self.op_type = op_type
        self.inputs = inputs


def test_coverage_gaps_cmd15():
    args = argparse.Namespace(
        model="test.onnx",
        skip_rules="a,b",
        prune_inputs="c",
        preserve_nodes="d",
        input_shape=["a:1,2", "b:a,b", "c:"],
        tensor_type=["a:float32"],
        check_n=3,
        custom_ops=["custom.py"],
        skip_fusions=True,
        skip_constant_folding=True,
        skip_shape_inference=True,
        skip_fuse_bn=True,
        dry_run=True,
        max_iterations=2,
        log_json=True,
        size_limit_mb=10,
        target_opset=12,
        strip_metadata=True,
        sort_value_info=True,
        diff_json=True,
        overwrite=True,
        output="out.onnx",
    )

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(
                nodes=[DummyNode("1", "1", ["old"]), DummyNode("2", "2", ["2"])],
                tensors={},
                inputs=[MagicMock(name="old", shape=(1, 2))],
                outputs=[],
            ),
        ),
        patch("onnx9000_cli.main.save_onnx"),
        patch("builtins.open"),
        patch("json.load", return_value=[{"action": "remove_node", "node_name": "1"}]),
        patch("os.path.exists", return_value=False),
        patch("importlib.util.spec_from_file_location", return_value=MagicMock(loader=MagicMock())),
        patch("importlib.util.module_from_spec", return_value=MagicMock()),
    ):
        with patch(
            "onnx9000_cli.main.simplify", return_value=MagicMock(nodes=[DummyNode("2", "2", ["2"])])
        ):
            try:
                simplify_cmd(args)
            except SystemExit:
                pass
            args.overwrite = False
            try:
                simplify_cmd(args)
            except SystemExit:
                pass

            args.output = None
            args.input_shape = None
            args.tensor_type = None
            args.skip_rules = None
            args.prune_inputs = None
            args.preserve_nodes = None
            args.check_n = None
            args.custom_ops = None
            try:
                simplify_cmd(args)
            except SystemExit:
                pass

            args.output = "out.onnx"
            with patch("os.path.exists", return_value=True):
                try:
                    simplify_cmd(args)
                except SystemExit:
                    pass
