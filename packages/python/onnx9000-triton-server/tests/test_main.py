from onnx9000_triton_server import TritonServer


def test_process():
    assert TritonServer().process("test") == "Triton Server processed test"
