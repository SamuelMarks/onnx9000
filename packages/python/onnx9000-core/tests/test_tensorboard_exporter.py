import os

from onnx9000.core.ir import Graph, Node
from onnx9000.core.tensorboard_exporter import export_tensorboard


def test_export_tensorboard(tmp_path):
    g = Graph("ResNet50")
    g.nodes.append(Node("Conv", name="conv1"))
    g.nodes.append(Node("Relu", name="relu1"))

    out_dir = str(tmp_path / "logs")
    filepath = export_tensorboard(g, out_dir)
    assert os.path.exists(filepath)
    with open(filepath, "rb") as f:
        content = f.read()
    assert b"node:layer_0/conv1,op:Conv" in content

    # Testing LLaMA graph
    g2 = Graph("LLaMA_v2")
    g2.nodes.append(Node("MatMul", name="matmul1"))
    filepath = export_tensorboard(g2, out_dir)
    assert os.path.exists(filepath)
    with open(filepath, "rb") as f:
        content = f.read()
    assert b"node:layer_0/matmul1,op:MatMul" in content
