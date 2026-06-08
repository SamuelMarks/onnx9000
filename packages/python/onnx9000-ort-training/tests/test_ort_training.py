from onnx9000_ort_training import ORTTraining


def test_ort_training():
    o = ORTTraining()
    assert o.process("test") == "ORT Training processed test"
