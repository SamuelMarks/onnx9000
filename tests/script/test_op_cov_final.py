from onnx9000.script.op import If, Loop, Scan
from onnx9000.script.builder import GraphBuilder


def test_script_op_zero_outputs():
    b = GraphBuilder("test")
    assert If(1, None, None, num_outputs=0) is None
    assert Loop(1, 1, None, num_outputs=0) is None
    assert Scan(1, None, num_outputs=0) is None
