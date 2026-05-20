from onnx9000_onnx_tool import ONNXTool


def test_process():
    assert ONNXTool().process("test") == "ONNX Tool processed test"
