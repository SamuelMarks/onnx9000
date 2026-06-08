from onnx9000_triton_server import TritonServer


def test_triton_server():
    t = TritonServer()
    assert t.process("test") == "Triton Server processed test"
