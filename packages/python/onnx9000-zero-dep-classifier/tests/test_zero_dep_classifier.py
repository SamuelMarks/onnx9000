from onnx9000_zero_dep_classifier import ZeroDepClassifier


def test_zero_dep():
    z = ZeroDepClassifier()
    assert z.process("test") == "Zero Dep Classifier processed test"
