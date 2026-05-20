from onnx9000_compile import run


def test_run():
    assert run() == "[compile] processed"
