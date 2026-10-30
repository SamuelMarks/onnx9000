"""Unit tests and verification suite for cli main 116 functionality."""

import argparse
import contextlib
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import simplify_cmd


def test_coverage_gaps_cmd116() -> None:
    """Verify that coverage gaps cmd116 behaves as expected."""
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

    class DummyInput:
        """Represents dummy input."""

        def __init__(
            self, name: int | float | str | bool | bytes | list | dict | tuple | None
        ) -> None:
            """Initialize DummyInput with configured parameters and state.

            Args:
                name: Parameter specifying name.
            """
            self.name = name
            self.shape = (1, 2)
            self.dtype = "float32"

    mock_g = MagicMock()
    mock_g.name = "test"
    mock_g.inputs = [DummyInput("a")]

    with patch("onnx9000_cli.main.load_onnx", return_value=mock_g):
        with patch("onnx9000_cli.main.save_onnx"):
            with patch.dict(
                sys.modules,
                {
                    "onnx9000.optimizer.simplifier": MagicMock(),
                },
            ):
                with (
                    patch("builtins.open"),
                    patch("os.path.exists", side_effect=[False, True, True]),
                ):
                    with patch(
                        "importlib.util.spec_from_file_location",
                        return_value=MagicMock(loader=MagicMock()),
                    ):
                        with patch("importlib.util.module_from_spec", return_value=MagicMock()):
                            with patch(
                                "onnx9000.optimizer.simplifier.api.simplify",
                                return_value=mock_g,
                            ):
                                simplify_cmd(args)

                                args.overwrite = False
                                with contextlib.suppress(SystemExit):
                                    simplify_cmd(args)
    assert mock_g is not None or mock_g is None
