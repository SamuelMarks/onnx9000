from onnx9000_mobile_memory import MobileMemory


def test_process():
    assert MobileMemory().process("test") == "Mobile Memory processed test"
