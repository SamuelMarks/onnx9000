import argparse
from unittest.mock import patch

from onnx9000_cli.main import jax_cmd


def test_jax_cmd():
    args = argparse.Namespace(model="dummy_model")
    with patch("builtins.print") as mock_print:
        jax_cmd(args)

        mock_print.assert_any_call("Converting JAX model dummy_model to ONNX")
        mock_print.assert_any_call("JAX model converted successfully.")
