import numpy as np
from onnx9000.backends.cpu.executor import CPUExecutionProvider
from onnx9000.backends.session import ExecutionContext, SessionOptions
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor


def test_executor_coverage():
    ep = CPUExecutionProvider({})
    g = Graph("g")
    g.add_node(
        Node("Binarizer", ["T1"], ["T2"], attributes={"threshold": Attribute("t", "FLOAT", 0.0)})
    )
    g.add_node(Node("MissingOp", ["T1"], ["T3"], attributes={}))
    g.add_node(
        Node("Binarizer", ["T2"], ["T4"], attributes={"threshold": Attribute("t", "FLOAT", 0.0)})
    )
    g.add_node(
        Node(
            "Binarizer", ["MISSING"], ["T5"], attributes={"threshold": Attribute("t", "FLOAT", 0.0)}
        )
    )
    t1 = Tensor("T1", (1,), DType.FLOAT32, data=np.array([1.0], dtype=np.float32))
    t1_nodata = Tensor("T1", (1,), DType.FLOAT32, data=None)
    assert ep._to_numpy(t1_nodata).shape == (1,)
    res = ep.execute(g, ExecutionContext(SessionOptions()), {"T1": t1})
    assert "T2" in res
    assert "T4" in res
    assert "T5" in res


def test_executor_bytes_dtype_mapping():
    ep = CPUExecutionProvider({})
    import struct

    t_bytes = Tensor("T", (1,), DType.FLOAT32, data=struct.pack("<f", 1.0))
    res = ep._to_numpy(t_bytes)
    assert res[0] == 1.0
    t_int = Tensor("T", (1,), DType.INT32, data=struct.pack("<i", 1))
    assert ep._to_numpy(t_int)[0] == 1
    t_int64 = Tensor("T", (1,), DType.INT64, data=struct.pack("<q", 1))
    assert ep._to_numpy(t_int64)[0] == 1
    t_bool = Tensor("T", (1,), DType.BOOL, data=struct.pack("<?", True))
    assert ep._to_numpy(t_bool)[0] == True
    g1 = Graph("g")
    g1.add_node(Node("MissingOp", [], [], {}))
    assert "MissingOp" not in ep.get_supported_nodes(g1)
    g2 = Graph("g")
    g2.add_node(Node("Add", [], [], {}))
    assert "Add" in ep.get_supported_nodes(g2)
    g3 = Graph("g")
    g3.add_node(Node("Binarizer", [], [], {}))
    assert "Binarizer" in ep.get_supported_nodes(g3)
    assert ep.allocate_tensors([]) is None


def test_executor_op_registry():
    ep = CPUExecutionProvider({})
    g = Graph("g")
    g.add_node(Node("Abs", ["T1"], ["T2"], attributes={}))
    t1 = Tensor("T1", (1,), DType.FLOAT32, data=np.array([-1.0], dtype=np.float32))
    res = ep.execute(g, ExecutionContext(SessionOptions()), {"T1": t1})
    assert res["T2"].data[0] == 1.0
