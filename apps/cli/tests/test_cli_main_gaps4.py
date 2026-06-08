import argparse
from unittest.mock import patch

from onnx9000_cli import main


def test_missing_commands():
    args = argparse.Namespace(model="test.onnx", src="test.onnx", dst="test.json", arch="test_arch")

    # commands without imports
    main.mlir_cmd(args)
    main.paddle2onnx_cmd(args)
    main.keras2onnx_cmd(args)
    main.skl2onnx_cmd(args)
    main.arena_cmd(args)

    # commands with imports (mocking import errors or successful imports if present)
    main.ort_training_cmd(args)
    main.olive_optimizer_cmd(args)
    main.triton_server_cmd(args)
    main.onnx_tool_cmd(args)
    main.mobile_memory_cmd(args)
    main.progressive_loading_cmd(args)
    main.new_model_arch_cmd(args)
    main.zero_dep_classifier_cmd(args)


test_missing_commands()
