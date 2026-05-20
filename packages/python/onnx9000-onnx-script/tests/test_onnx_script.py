from onnx9000_onnx_script import run


def test_run():
    assert run() == "[onnx-script] processed"
