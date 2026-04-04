"""Module docstring."""


def test_profiler_coverage():
    """Docstring for D103."""
    import pytest
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Constant, Graph, Node, Tensor
    from onnx9000.core.profiler import dtype_size, get_attr, profile

    assert dtype_size(DType.FLOAT32) == 4
    assert dtype_size(None) == 4

    g = Graph("test")
    x = Tensor("x", shape=[1, 3, 224, 224], dtype=DType.FLOAT32)
    w = Tensor("w", shape=[64, 3, 7, 7], dtype=DType.FLOAT32)
    w.is_initializer = True
    b = Tensor("b", shape=[64], dtype=DType.FLOAT32)
    y = Tensor("y", shape=[1, 64, 112, 112], dtype=DType.FLOAT32)

    n = Node("Conv", inputs=["x", "w", "b"], outputs=["y"])
    n.attributes["kernel_shape"] = type("obj", (object,), {"name": "kernel_shape", "value": [7, 7]})
    n.attributes["group"] = type("obj", (object,), {"name": "group", "value": 1})
    g.nodes.append(n)

    g.tensors = {"x": x, "w": w, "b": b, "y": y}
    g.inputs.append(x)
    g.outputs.append(y)

    report = profile(g)

    assert report.total_macs > 0
    assert report.total_flops > 0
    assert report.total_params > 0
