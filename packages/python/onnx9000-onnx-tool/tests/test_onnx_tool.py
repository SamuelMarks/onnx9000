from onnx9000_onnx_tool import ONNXTool


def test_onnx_tool():
    o = ONNXTool()
    assert o.process("test") == "ONNX Tool processed test"
