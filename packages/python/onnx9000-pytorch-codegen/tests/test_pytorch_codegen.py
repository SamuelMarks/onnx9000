from onnx9000_pytorch_codegen import run


def test_run():
    assert run() == "[pytorch-codegen] processed"
