from onnx9000.toolkit.script.builder import GraphBuilder
from onnx9000.toolkit.script.op import If, Loop, Scan


def test_script_op_zero_outputs() -> None:
    GraphBuilder("test")
    assert If(1, None, None, num_outputs=0) is None
    assert Loop(1, 1, None, num_outputs=0) is None
    assert Scan(1, None, num_outputs=0) is None
