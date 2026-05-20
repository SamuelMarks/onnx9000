from onnx9000_onnx_checker import run


def test_run():
    assert run() == "[onnx-checker] processed"
