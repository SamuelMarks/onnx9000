from onnx9000_new_model_arch import NewModelArch


def test_process():
    assert NewModelArch().process("test") == "New Model Arch processed test"
