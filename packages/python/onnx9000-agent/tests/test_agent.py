from onnx9000_agent import run


def test_run():
    assert run() == "[agent] processed"
