from onnx9000_ort_training import ORTTraining


def test_process():
    assert ORTTraining().process("test") == "ORT Training processed test"
