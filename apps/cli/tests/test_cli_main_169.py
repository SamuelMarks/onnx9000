import argparse
import os

from onnx9000_cli.main import convert_cmd


def test_convert_flax_msgpack(tmp_path):
    model_path = tmp_path / "model.msgpack"
    out_path = tmp_path / "model_out.onnx"

    # Valid msgpack for {"params": {}}
    model_path.write_bytes(b"\x81\xa6params\x80")

    args = argparse.Namespace(
        src=str(model_path), from_fmt="flax", to_fmt="onnx", output=str(out_path), weights=None
    )

    convert_cmd(args)
    assert os.path.exists(out_path)


def test_convert_flax_json(tmp_path):
    model_path = tmp_path / "model.json"
    out_path = tmp_path / "model_out.onnx"

    # Fallback to JSON
    import json

    model_path.write_text(json.dumps({"params": {}}))

    args = argparse.Namespace(
        src=str(model_path), from_fmt="flax", to_fmt="onnx", output=str(out_path), weights=None
    )

    convert_cmd(args)
    assert os.path.exists(out_path)
