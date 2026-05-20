from onnx9000_zero_dep_classifier import ZeroDepClassifier


def test_process():
    assert ZeroDepClassifier().process("test") == "Zero Dep Classifier processed test"
