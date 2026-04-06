"""Tests for the ONNX9000 SDK CLI bindings."""


def test_sdk_cli_bindings():
    """Verify that autograd, tvm, and diffusers are exposed in the CLI."""
    import argparse
    from unittest.mock import patch

    import onnx9000_cli.main as main

    args = argparse.Namespace(
        model="test.onnx",
        output=None,
    )

    with (
        patch("onnx9000_cli.main.load_onnx") as mock_load,
        patch("onnx9000_cli.main.save_onnx") as mock_save,
        patch("onnx9000.toolkit.training.autograd.compiler.AutogradEngine") as mock_engine_cls,
    ):
        mock_graph = "mock_fwd_graph"
        mock_load.return_value = mock_graph
        mock_bwd_graph = "mock_bwd_graph"
        mock_engine = mock_engine_cls.return_value
        mock_engine.build_backward_graph.return_value = mock_bwd_graph

        main.autograd_cmd(args)

        mock_load.assert_called_with("test.onnx")
        mock_engine.build_backward_graph.assert_called_with(mock_graph)
        mock_save.assert_called_with(mock_bwd_graph, "test_bw.onnx")

    with patch("onnx9000.tvm.build_module.build") as mock_build:
        args.target = "llvm"
        main.tvm_cmd(args)
        mock_build.assert_called()

    with patch("onnx9000_diffusers.pipeline.DiffusionPipeline") as mock_pipe:
        args.diffusers_command = "export"
        args.model_id = "test"
        main.diffusers_cmd(args)
        mock_pipe.from_pretrained.assert_called_with("test")

        args.diffusers_command = None
        main.diffusers_cmd(args)

    with patch("onnx9000.tensorrt.builder.Builder") as mock_trt:
        main.tensorrt_cmd(args)
        mock_trt.assert_called()

    args_onnx2tf = argparse.Namespace(
        input="test.onnx",
        output="test.tflite",
        keep_nchw=True,
        int8=True,
        fp16=False,
        batch=1,
        disable_optimization=True,
        external_weights="weights.bin",
        progress=True,
        micro=True,
    )
    with patch("onnx9000.tflite_exporter.cli.main") as mock_onnx2tf_main:
        main.onnx2tf_cmd(args_onnx2tf)
        mock_onnx2tf_main.assert_called()

    args_onnx2tf_2 = argparse.Namespace(
        input="test.onnx",
        output=None,
        keep_nchw=False,
        int8=False,
        fp16=True,
        batch=None,
        disable_optimization=False,
        external_weights=None,
        progress=False,
        micro=False,
    )
    with patch("onnx9000.tflite_exporter.cli.main") as mock_onnx2tf_main_2:
        main.onnx2tf_cmd(args_onnx2tf_2)
        mock_onnx2tf_main_2.assert_called()
