from onnx9000_new_model_arch import NewModelArch


def test_process():
    n = NewModelArch()
    assert n.process("test") == "New Model Arch processed test"
