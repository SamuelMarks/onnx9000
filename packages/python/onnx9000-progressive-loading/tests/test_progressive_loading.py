from onnx9000_progressive_loading import ProgressiveLoading


def test_progressive_loading():
    p = ProgressiveLoading()
    assert p.process("test") == "Progressive Loading processed test"
