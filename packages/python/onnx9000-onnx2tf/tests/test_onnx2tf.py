from onnx9000_onnx2tf import run


def test_run():
    assert run() == "[onnx2tf] processed"
