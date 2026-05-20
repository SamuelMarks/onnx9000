from onnx9000_progressive_loading import ProgressiveLoading


def test_process():
    assert ProgressiveLoading().process("test") == "Progressive Loading processed test"
