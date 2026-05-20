from onnx9000_onnx2c import run


def test_run():
    assert run() == "[onnx2c] processed"
