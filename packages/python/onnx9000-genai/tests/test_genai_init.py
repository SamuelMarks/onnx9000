from onnx9000_genai import run


def test_run():
    assert run() == "[genai] processed"
