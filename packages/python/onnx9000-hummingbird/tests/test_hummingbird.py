from onnx9000_hummingbird import run


def test_run():
    assert run() == "[hummingbird] processed"
