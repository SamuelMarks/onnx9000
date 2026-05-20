from onnx9000_c_compiler import run


def test_run():
    assert run() == "[c_compiler] processed"
