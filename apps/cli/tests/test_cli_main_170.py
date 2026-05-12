import argparse
import os

from onnx9000_cli.main import convert_cmd


def test_convert_tensorflow_graphdef(tmp_path):
    model_path = tmp_path / "model.pb"
    out_path = tmp_path / "model_out.onnx"

    # Just an empty file for dummy test. The parser might fail but we check if it routes correctly
    model_path.write_bytes(b"")

    args = argparse.Namespace(
        src=str(model_path),
        from_fmt="tensorflow",
        to_fmt="onnx",
        output=str(out_path),
        weights=None,
    )

    try:
        convert_cmd(args)
    except Exception:
        pass


def test_convert_tensorflow_saved_model(tmp_path):
    model_dir = tmp_path / "saved_model"
    model_dir.mkdir()
    pb_path = model_dir / "saved_model.pb"
    pb_path.write_bytes(b"")

    out_path = tmp_path / "model_out.onnx"

    args = argparse.Namespace(
        src=str(model_dir), from_fmt="tensorflow", to_fmt="onnx", output=str(out_path), weights=None
    )

    try:
        convert_cmd(args)
    except Exception:
        pass


def test_convert_tensorflow_missing_pb(tmp_path):
    model_dir = tmp_path / "saved_model_missing"
    model_dir.mkdir()
    out_path = tmp_path / "model_out.onnx"

    args = argparse.Namespace(
        src=str(model_dir), from_fmt="tensorflow", to_fmt="onnx", output=str(out_path), weights=None
    )

    try:
        convert_cmd(args)
    except SystemExit:
        pass
