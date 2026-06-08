from onnx9000_mmdnn import run


def test_run():
    assert run() == "[mmdnn] processed"
