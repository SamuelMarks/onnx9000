from onnx9000_mlir_lowering import run


def test_run():
    assert run() == "[mlir-lowering] processed"
