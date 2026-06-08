from onnx9000_mobile_memory import MobileMemory


def test_process():
    m = MobileMemory()
    assert m.process("test") == "Mobile Memory processed test"
