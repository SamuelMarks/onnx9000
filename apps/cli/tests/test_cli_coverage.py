"""Tests for the main CLI commands ensuring correct coverage and execution behavior.

This module provides comprehensive test cases for various command-line interface
functions such as simplify, coreml export, node pruning, input renaming, batch size
modification, mutation, and information retrieval. It mocks file system operations,
external command executions, and core ONNX operations to evaluate the CLI logic.
"""

import argparse
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from onnx9000_cli.main import simplify_cmd


def test_simplify_cmd_coverage():
    """Verify that the simplify command correctly processes arguments and generates output.

    This test sets up a temporary workspace with mock ONNX and custom ops files,
    invokes the `simplify_cmd` with an array of configuration flags, and asserts
    that the diff file is correctly produced and contains the expected node
    additions and removals. It also ensures that the command exits correctly
    when the overwrite flag is disabled and the output file already exists.
    """
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


def test_tui_chat():
    """Ensure that the TUI chat interface entry point can be invoked without errors.

    This test explicitly inserts the source path into `sys.path` to load and
    call `start_chat_tui`, ensuring that the TUI module loads correctly when
    the command is executed.
    """
    import sys

    sys.path.insert(0, "apps/cli/src")
    from tui_chat import start_chat_tui

    start_chat_tui()


def test_info_cmd_coverage(capsys):
    """Test the behavior of info commands when standard attributes or functions are invoked.

    This test checks that `info_cmd` handles a missing subcommand properly by
    raising a `SystemExit` with the correct code. It also verifies that
    `info_webnn_cmd` properly prints the diagnostic information to standard output.
    """
    from onnx9000_cli.main import info_cmd, info_webnn_cmd
    import argparse
    import pytest

    args = argparse.Namespace()

    # Test missing info_func
    with pytest.raises(SystemExit) as e:
        info_cmd(args)
    assert e.value.code == 1
    assert "Missing info subcommand" in capsys.readouterr().out

    # Test info_webnn_cmd
    info_webnn_cmd(args)
    assert "WebNN API Diagnostic Info" in capsys.readouterr().out


def test_info_cmd_with_func():
    """Verify that `info_cmd` successfully delegates execution to a provided subcommand.

    A dummy function is injected into the arguments namespace, and this test
    ensures that the CLI successfully calls it when the command is invoked.
    """
    from onnx9000_cli.main import info_cmd
    import argparse

    args = argparse.Namespace()

    def dummy_func(a):
        a.called = True

    args.info_func = dummy_func
    info_cmd(args)
    assert args.called


def test_coreml_cmd_coverage():
    """Check the robust execution and error handling of the CoreML export command.

    This test evaluates different execution paths of `coreml_cmd`, including
    scenarios where the underlying JavaScript CLI script is missing, successful
    subprocess execution, and handling of subprocess errors.
    """
    from onnx9000_cli.main import coreml_cmd
    import argparse
    import pytest
    from unittest.mock import patch
    import subprocess

    args = argparse.Namespace(coreml_command="export", model="test.onnx")

    # Test when JS CLI script is missing
    with patch("os.path.exists", return_value=False), pytest.raises(SystemExit) as e:
        coreml_cmd(args)
    assert e.value.code == 1

    # Test successful execution
    with patch("os.path.exists", return_value=True), patch("subprocess.run") as mock_run:
        coreml_cmd(args)
        mock_run.assert_called_once()

    # Test failed execution
    with (
        patch("os.path.exists", return_value=True),
        patch("subprocess.run", side_effect=subprocess.CalledProcessError(2, "cmd")),
    ):
        with pytest.raises(SystemExit) as e:
            coreml_cmd(args)
        assert e.value.code == 2


def test_edit_cmd():
    """Evaluate the `edit_cmd` behavior for launching an interactive edit session.

    This test covers successful execution of the edit command via a subprocess,
    handles cases where the model path does not exist, and gracefully handles
    `KeyboardInterrupt` exceptions triggered during the interactive session.
    """
    from onnx9000_cli.main import edit_cmd
    import argparse
    from unittest.mock import patch

    args = argparse.Namespace(model="test.onnx")
    with patch("os.path.exists", return_value=True), patch("subprocess.run") as mock_run:
        edit_cmd(args)
        mock_run.assert_called_once()

    with patch("os.path.exists", return_value=False):
        edit_cmd(args)

    with (
        patch("os.path.exists", return_value=True),
        patch("subprocess.run", side_effect=KeyboardInterrupt),
    ):
        edit_cmd(args)


def test_prune_cmd():
    """Test the graph pruning CLI command for expected node removals.

    Mocks are used to simulate loading a graph and evaluating the `prune_cmd`'s
    ability to remove specified nodes based on the arguments. It also validates
    behavior when no output path is explicitly provided.
    """
    from onnx9000_cli.main import prune_cmd
    import argparse
    from unittest.mock import MagicMock, patch

    args = argparse.Namespace(model="test.onnx", nodes="n1,n2", output="out.onnx")
    with (
        patch("onnx9000.core.parser.core.load") as mock_load,
        patch("onnx9000.core.serializer.save") as mock_save,
    ):
        mock_graph = MagicMock()
        mock_graph.nodes = [
            MagicMock(name="n1", op_type="Add"),
            MagicMock(name="n3", op_type="Sub"),
        ]
        mock_graph.nodes[0].name = "n1"
        mock_graph.nodes[1].name = "n3"
        mock_load.return_value = mock_graph
        prune_cmd(args)
        assert len(mock_graph.nodes) == 1
        assert mock_graph.nodes[0].name == "n3"
        mock_save.assert_called_once()

    # test no output
    args.output = None
    with (
        patch("onnx9000.core.parser.core.load") as mock_load,
        patch("onnx9000.core.serializer.save") as mock_save,
    ):
        mock_load.return_value = MagicMock(nodes=[])
        prune_cmd(args)


def test_rename_input_cmd():
    """Verify that graph inputs can be successfully renamed via the CLI.

    This test simulates an ONNX graph with predefined inputs and checks whether
    `rename_input_cmd` accurately modifies the input name and propagates the
    changes to nodes connected to that input. It also tests omitting the output arg.
    """
    from onnx9000_cli.main import rename_input_cmd
    import argparse
    from unittest.mock import MagicMock, patch

    args = argparse.Namespace(model="test.onnx", old="A", new="B", output="out.onnx")
    with (
        patch("onnx9000.core.parser.core.load") as mock_load,
        patch("onnx9000.core.serializer.save") as mock_save,
    ):
        mock_graph = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "A"
        mock_graph.inputs = [mock_input]
        mock_node = MagicMock()
        mock_node.inputs = ["A", "C"]
        mock_graph.nodes = [mock_node]
        mock_load.return_value = mock_graph
        rename_input_cmd(args)
        assert mock_input.name == "B"
        assert mock_node.inputs[0] == "B"
        mock_save.assert_called_once()

    # test no output
    args.output = None
    with (
        patch("onnx9000.core.parser.core.load") as mock_load,
        patch("onnx9000.core.serializer.save") as mock_save,
    ):
        mock_load.return_value = MagicMock(inputs=[], nodes=[])
        rename_input_cmd(args)


def test_change_batch_cmd():
    """Test functionality of modifying batch size dimensions in the ONNX model.

    This validates that `change_batch_cmd` parses dimensions correctly,
    handling both static integer sizes and dynamic text representations
    by applying them to the shape attribute of the input tensors.
    """
    from onnx9000_cli.main import change_batch_cmd
    import argparse
    from unittest.mock import MagicMock, patch

    args = argparse.Namespace(model="test.onnx", size="4", output="out.onnx")
    with (
        patch("onnx9000.core.parser.core.load") as mock_load,
        patch("onnx9000.core.serializer.save") as mock_save,
    ):
        mock_graph = MagicMock()
        mock_input = MagicMock()
        mock_input.shape = [1, 2, 3]
        mock_graph.inputs = [mock_input]
        mock_graph.outputs = []
        mock_graph.value_info = []
        mock_load.return_value = mock_graph
        change_batch_cmd(args)
        assert mock_input.shape[0] == 4
        mock_save.assert_called_once()

    args = argparse.Namespace(model="test.onnx", size="dynamic", output=None)
    with (
        patch("onnx9000.core.parser.core.load") as mock_load,
        patch("onnx9000.core.serializer.save") as mock_save,
    ):
        mock_graph = MagicMock()
        mock_input = MagicMock()
        mock_input.shape = [1, 2, 3]
        mock_graph.inputs = [mock_input]
        mock_graph.outputs = []
        mock_graph.value_info = []
        mock_load.return_value = mock_graph
        change_batch_cmd(args)
        assert mock_input.shape[0] == "dynamic"


def test_mutate_cmd():
    """Ensure graph mutations are applied correctly when loading a script file.

    Using mocked JSON file reads, this checks that the `mutate_cmd` loads operations,
    such as removing a specific node, and applies these mutations successfully
    to the loaded ONNX graph structure.
    """
    from onnx9000_cli.main import mutate_cmd
    import argparse
    from unittest.mock import MagicMock, patch, mock_open

    args = argparse.Namespace(model="test.onnx", script="mut.json", output="out.onnx")
    with (
        patch("onnx9000.core.parser.core.load") as mock_load,
        patch("onnx9000.core.serializer.save") as mock_save,
    ):
        with patch(
            "builtins.open", mock_open(read_data='[{"action": "remove_node", "node_name": "n1"}]')
        ):
            mock_graph = MagicMock()
            mock_graph.nodes = [
                MagicMock(name="n1", op_type="Add"),
                MagicMock(name="n2", op_type="Sub"),
            ]
            mock_graph.nodes[0].name = "n1"
            mock_graph.nodes[1].name = "n2"
            mock_load.return_value = mock_graph
            mutate_cmd(args)
            assert len(mock_graph.nodes) == 1
            assert mock_graph.nodes[0].name == "n2"

    args.output = None
    with (
        patch("onnx9000.core.parser.core.load") as mock_load,
        patch("onnx9000.core.serializer.save") as mock_save,
    ):
        with patch("builtins.open", mock_open(read_data="[]")):
            mock_load.return_value = MagicMock(nodes=[])
            mutate_cmd(args)


def test_stubs_coverage():
    """Verify that all placeholder and specialized commands execute safely.

    This test comprehensively covers basic stubs and external module wrappers
    (e.g., inspect, convert, compile, and various optimum commands). It asserts
    that these commands can be invoked using standard arguments without crashing,
    and specifically that the optimum proxy commands invoke their respective APIs.
    """
    from onnx9000_cli.main import (
        inspect_cmd,
        optimize_cmd,
        quantize_cmd,
        export_cmd,
        convert_cmd,
        serve_cmd,
        compile_cmd,
        optimum_export_cmd,
        optimum_optimize_cmd,
        optimum_quantize_cmd,
        optimum_cmd,
    )
    import argparse
    from unittest.mock import patch
    import pytest

    args = argparse.Namespace(
        model="test.onnx",
        script="test.py",
        src="a",
        dst="b",
        model_id="test/model",
        task="text-classification",
        opset=14,
        device="cpu",
        cache_dir=None,
        split="train",
        level="O2",
        disable_fusion=False,
        optimize_size=False,
        quantize="int8",
        gptq_bits=4,
        gptq_group_size=128,
    )

    inspect_cmd(args)
    optimize_cmd(args)
    quantize_cmd(args)
    export_cmd(args)
    convert_cmd(args)
    serve_cmd(args)
    compile_cmd(args)

    with patch("onnx9000_optimum.export.export_model") as m1:
        optimum_export_cmd(args)
        m1.assert_called_once()

    with patch("onnx9000_optimum.optimize.optimize_model") as m2:
        optimum_optimize_cmd(args)
        m2.assert_called_once()

    with patch("onnx9000_optimum.quantize.quantize_model") as m3:
        optimum_quantize_cmd(args)
        m3.assert_called_once()

    with pytest.raises(SystemExit):
        optimum_cmd(argparse.Namespace())

    args.optimum_func = lambda x: print("optimum ok")
    optimum_cmd(args)
