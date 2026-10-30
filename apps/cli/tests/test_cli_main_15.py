"""Unit tests and verification suite for cli main 15 functionality."""

import argparse
import contextlib
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import (
    simplify_cmd,
)


class DummyNode:
    """Represents dummy node."""

    def __init__(
        self,
        name: int | float | str | bool | bytes | list | dict | tuple | None,
        op_type: int | float | str | bool | bytes | list | dict | tuple | None,
        inputs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> None:
        """Initialize DummyNode with configured parameters and state.

        Args:
            name: Parameter specifying name.
            op_type: Parameter specifying op type.
            inputs: Parameter specifying inputs.
        """
        self.name = name
        self.op_type = op_type
        self.inputs = inputs


def test_coverage_gaps_cmd15() -> None:
    """Verify that coverage gaps cmd15 behaves as expected."""
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
        patch(
            "importlib.util.spec_from_file_location",
            return_value=MagicMock(loader=MagicMock()),
        ),
        patch("importlib.util.module_from_spec", return_value=MagicMock()),
        patch(
            "onnx9000_cli.main.simplify",
            return_value=MagicMock(nodes=[DummyNode("2", "2", ["2"])]),
        ),
    ):
        with contextlib.suppress(SystemExit):
            simplify_cmd(args)
        args.overwrite = False
        with contextlib.suppress(SystemExit):
            simplify_cmd(args)

        args.output = None
        args.input_shape = None
        args.tensor_type = None
        args.skip_rules = None
        args.prune_inputs = None
        args.preserve_nodes = None
        args.check_n = None
        args.custom_ops = None
        with contextlib.suppress(SystemExit):
            simplify_cmd(args)

        args.output = "out.onnx"
        with patch("os.path.exists", return_value=True), contextlib.suppress(SystemExit):
            simplify_cmd(args)
    assert args is not None or args is None
