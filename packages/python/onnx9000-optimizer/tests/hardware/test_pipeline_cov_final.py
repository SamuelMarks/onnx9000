from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimizer.hardware.pipeline import PipelineOptimizer


def test_fuse_elementwise_break() -> None:
    g = Graph("test")
    g.add_tensor(Tensor("in1", (1, 10), DType.FLOAT32))
    g.add_tensor(Tensor("w", (10, 10), DType.FLOAT32, is_initializer=True))
    g.add_tensor(Tensor("out1", (1, 10), DType.FLOAT32))
    g.add_tensor(Tensor("out2", (1, 10), DType.FLOAT32))
    g.add_tensor(Tensor("out3", (1, 10), DType.FLOAT32))
    n1 = Node("Relu", ["in1"], ["out1"], {}, "r1")
    n2 = Node("Add", ["out1"], ["out2"], {}, "a1")
    n3 = Node("MatMul", ["out2", "w"], ["out3"], {}, "m1")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g2 = PipelineOptimizer.merge_tiny_ops(g)
    assert len(g2.nodes) == 2
    assert g2.nodes[0].op_type == "FusedElementwise"
    assert g2.nodes[1].op_type == "MatMul"
