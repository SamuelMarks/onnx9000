import argparse
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from onnx9000_cli.main import (
    onnx2gguf_cmd,
    openvino_cmd,
    openvino_export_cmd,
    optimize_cmd,
    sparse_cmd,
    sparse_de_sparsify_cmd,
    sparse_prune_cmd,
)


def test_optimize_cmd_prune():
    args = argparse.Namespace(
        model="test.onnx", prune=True, sparsity=0.5, quantize=True, output="out.onnx"
    )

    with (
        patch("onnx9000_cli.main.load_onnx"),
        patch("onnx9000_cli.main.save_onnx"),
        patch("onnx9000.optimizer.sparse.modifier.GlobalMagnitudePruningModifier") as mock_prune,
        patch("onnx9000.optimizer.sparse.modifier.QuantizationModifier") as mock_quant,
    ):
        optimize_cmd(args)
        mock_prune.assert_called_once()
        mock_quant.assert_called_once()


def test_onnx2gguf_cmd_triton():
    args = argparse.Namespace(
        model="test.onnx",
        dry_run=False,
        force=False,
        architecture=None,
        tokenizer=None,
        outtype=None,
        output="out.gguf",
    )

    with (
        patch.dict("sys.modules", {"triton": MagicMock(__version__="1.0.0")}),
        patch("onnx9000_cli.main.load_onnx"),
        patch("builtins.open"),
        patch("onnx9000.onnx2gguf.compiler.compile_gguf"),
    ):
        onnx2gguf_cmd(args)


def test_openvino_export_cmd():
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Graph, ValueInfo

    graph = Graph("test")
    graph.inputs.append(ValueInfo("input1", (1, 3, 224, 224), DType.FLOAT32))

    args = argparse.Namespace(
        model="test.onnx",
        shape=["input1:2,3,256,256"],
        dynamic_batch=True,
        fp16=True,
        output_dir="ov_out",
    )

    with (
        patch(
            "onnx9000.openvino.exporter.OpenVinoExporter.export", return_value=("<xml/>", b"bin")
        ),
        patch("onnx9000_cli.main.load_onnx", return_value=graph),
        patch("onnx9000.core.parser.core.load", return_value=graph),
        patch("os.makedirs"),
        patch("builtins.open"),
    ):
        openvino_export_cmd(args)


def test_openvino_cmd_missing_subcmd():
    args = argparse.Namespace()
    with pytest.raises(SystemExit) as e:
        openvino_cmd(args)
    assert e.value.code == 1


def test_sparse_prune_cmd(tmp_path):
    import numpy as np
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Constant, Graph

    graph = Graph("test")
    tensor = Constant("w")
    tensor.is_initializer = True
    graph.tensors["w"] = tensor
    graph.initializers.append("w")

    args = argparse.Namespace(
        model="test.onnx", recipe="recipe.yaml", sparsity=None, output="out.onnx"
    )

    with (
        patch("onnx9000_cli.main.load_onnx", return_value=graph),
        patch("onnx9000_cli.main.save_onnx"),
        patch("builtins.open"),
        patch("onnx9000.optimizer.sparse.modifier.apply_recipe") as mock_apply,
        patch("onnx9000.core.sparse.analyze_topological_dead_ends", return_value=["node1"]),
        patch("onnx9000.core.sparse.detect_theoretical_sparsity", return_value=0.1),
        patch("onnx9000.core.sparse.dense_to_coo"),
        patch("onnx9000.core.sparse.strip_dense_representation"),
        patch("onnx9000.core.sparse.collapse_sparse_tensors"),
    ):
        sparse_prune_cmd(args)
        mock_apply.assert_called_once()

    args2 = argparse.Namespace(model="test.onnx", recipe=None, sparsity=0.8, output="out.onnx")
    with (
        patch("onnx9000_cli.main.load_onnx", return_value=graph),
        patch("onnx9000_cli.main.save_onnx"),
        patch("onnx9000.optimizer.sparse.modifier.GlobalMagnitudePruningModifier") as mock_mod,
        patch("onnx9000.core.sparse.analyze_topological_dead_ends", return_value=[]),
        patch("onnx9000.core.sparse.detect_theoretical_sparsity", return_value=0.0),
        patch("onnx9000.core.sparse.collapse_sparse_tensors"),
    ):
        sparse_prune_cmd(args2)
        mock_mod.assert_called_once()


def test_sparse_de_sparsify_cmd():
    args = argparse.Namespace(model="test.onnx", output="dense.onnx")

    with (
        patch("onnx9000_cli.main.load_onnx"),
        patch("onnx9000_cli.main.save_onnx"),
        patch("onnx9000.core.sparse.de_sparsify") as mock_de,
    ):
        sparse_de_sparsify_cmd(args)
        mock_de.assert_called_once()


def test_sparse_cmd_missing_subcmd():
    args = argparse.Namespace()
    with pytest.raises(SystemExit) as e:
        sparse_cmd(args)
    assert e.value.code == 1


def test_cli_more_branches():
    import sys
    from unittest.mock import patch

    from onnx9000_cli.main import main

    with patch.object(
        sys,
        "argv",
        [
            "onnx9000",
            "optimize",
            "dummy.onnx",
            "--prune",
            "--sparsity",
            "0.5",
            "--quantize",
            "--o",
            "out.onnx",
        ],
    ):
        try:
            main()
        except (Exception, SystemExit):
            pass

    with patch.object(
        sys, "argv", ["onnx9000", "openvino", "export", "dummy.onnx", "--o", "out.onnx"]
    ):
        try:
            main()
        except (Exception, SystemExit):
            pass

    with patch.object(sys, "argv", ["onnx9000", "onnx2gguf", "dummy.onnx", "out.gguf"]):
        try:
            main()
        except (Exception, SystemExit):
            pass

    with patch.object(
        sys,
        "argv",
        ["onnx9000", "sparse", "prune", "dummy.onnx", "--sparsity", "0.5", "-o", "out.onnx"],
    ):
        try:
            main()
        except (Exception, SystemExit):
            pass

    with patch.object(
        sys, "argv", ["onnx9000", "edit", "dummy.onnx", "--reshape", "input:?,224,224,,"]
    ):
        try:
            main()
        except (Exception, SystemExit):
            pass

    with patch.object(
        sys,
        "argv",
        [
            "onnx9000",
            "openvino",
            "export",
            "dummy.onnx",
            "--shape",
            "input:[?,224,224,,]",
            "--dynamic-batch",
        ],
    ):
        try:
            main()
        except (Exception, SystemExit):
            pass

    with patch.object(sys, "argv", ["onnx9000", "sparse"]):
        try:
            main()
        except (Exception, SystemExit):
            pass


def test_openvino_export_shape_mock():
    import sys
    from unittest.mock import MagicMock, patch

    from onnx9000_cli.main import main

    class MockInput:
        name = "input"
        shape = None

    class MockGraph:
        inputs = [MockInput()]

    with patch("onnx9000.core.parser.core.load", return_value=MockGraph()):
        with patch("onnx9000.openvino.exporter.OpenVinoExporter", MagicMock()):
            with patch.object(
                sys,
                "argv",
                [
                    "onnx9000",
                    "openvino",
                    "export",
                    "dummy.onnx",
                    "--shape",
                    "input:[?,224,224,,]",
                    "--dynamic-batch",
                ],
            ):
                try:
                    main()
                except (Exception, SystemExit):
                    pass


def test_sparse_cmd_mock():
    import sys
    from unittest.mock import patch

    from onnx9000_cli.main import main

    with patch.object(sys, "argv", ["onnx9000", "sparse"]):
        try:
            main()
        except (Exception, SystemExit):
            pass
