from onnx9000_simplify import run


def test_run():
    assert run() == "[simplify] processed"
