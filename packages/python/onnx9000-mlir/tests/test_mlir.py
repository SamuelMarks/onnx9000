from onnx9000_mlir import run


def test_run():
    assert run() == "[mlir] processed"
