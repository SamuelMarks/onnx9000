from onnx9000_olive_optimizer import OliveOptimizer


def test_process():
    assert OliveOptimizer().process("test") == "Olive Optimizer processed test"
