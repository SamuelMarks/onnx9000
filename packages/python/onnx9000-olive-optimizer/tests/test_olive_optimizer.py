from onnx9000_olive_optimizer import OliveOptimizer


def test_olive_optimizer():
    o = OliveOptimizer()
    assert o.process("test") == "Olive Optimizer processed test"
