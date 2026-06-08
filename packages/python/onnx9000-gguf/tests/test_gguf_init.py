from onnx9000_gguf import run


def test_run():
    assert run() == "[gguf] processed"
