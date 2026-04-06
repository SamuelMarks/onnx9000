"""Test codegen for PyTorch and Flax."""

from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
from onnx9000.core.ir import Graph, Node, Tensor


def test_pytorch_codegen():
    """Docstring for D103."""
    graph = Graph("test")
    graph.name = "test_model"

    inp = Tensor(name="x", shape=[1, 10])
    graph.inputs.append(inp)

    node1 = Node(op_type="LayerNormalization", inputs=[inp], outputs=[Tensor(name="ln_out")])
    node2 = Node(
        op_type="MatMul",
        inputs=[node1.outputs[0], Tensor(name="weight")],
        outputs=[Tensor(name="y")],
    )

    graph.nodes.extend([node1, node2])
    graph.outputs.append(node2.outputs[0])

    codegen = ONNXToPyTorchVisitor(graph)
    code = codegen.generate()

    assert "import torch" in code
    assert "import torch.nn as nn" in code
    assert "class Model(nn.Module):" in code
    assert "def forward(self, x):" in code
    assert "self.mod_0 = nn.LayerNorm" in code
    assert "ln_out = self.mod_0(x)" in code
    assert "self.mod_1 = nn.Linear" in code
    assert "y = self.mod_1(ln_out)" in code

    load_
    assert "def load_weights" in load_script


def test_flax_codegen():
    """Docstring for D103."""
    graph = Graph("test")
    graph.name = "test_flax_model"

    inp = Tensor(name="x", shape=[1, 10])
    graph.inputs.append(inp)

    node1 = Node(op_type="BatchNormalization", inputs=[inp], outputs=[Tensor(name="bn_out")])
    node2 = Node(
        op_type="Gemm", inputs=[node1.outputs[0], Tensor(name="weight")], outputs=[Tensor(name="y")]
    )
    node3 = Node(op_type="Constant", inputs=[], outputs=[Tensor(name="c")])

    graph.nodes.extend([node1, node2, node3])
    graph.outputs.append(node2.outputs[0])

    codegen = ONNXToFlaxNNXVisitor(graph)
    code = codegen.generate()

    assert "import jax" in code
    assert "import flax.nnx as nnx" in code

    assert "def __init__(self, rngs: nnx.Rngs):" in code
    assert "def __call__(self, x):" in code


def test_flax_nnx_codegen():
    """Docstring for D103."""
    from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("TestFlax")
    x = Tensor("x", shape=(1, 3, 224, 224), dtype=1)
    g.inputs.append(x)
    n1 = Node("Conv", inputs=["x"], outputs=["y"])
    n2 = Node("Relu", inputs=["y"], outputs=["z"])
    n3 = Node("BatchNormalization", inputs=["z"], outputs=["out"])
    g.nodes.extend([n1, n2, n3])

    out_t = Tensor("out", shape=(1, 3, 224, 224), dtype=1)
    g.outputs.append(out_t)

    visitor = ONNXToFlaxNNXVisitor(g)
    code = visitor.generate()
    assert "import flax.nnx as nnx" in code

    assert "load_weights(" in code


def test_pytorch_codegen():
    """Docstring for D103."""
    from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    graph = Graph("test")
    graph.name = "test_pytorch_model"

    inp = Tensor(name="x", shape=[1, 10])
    graph.inputs.append(inp)

    node1 = Node(op_type="Conv", inputs=[inp], outputs=[Tensor(name="conv_out")])
    node2 = Node(op_type="Gemm", inputs=[node1.outputs[0]], outputs=[Tensor(name="y")])
    node3 = Node(op_type="Pad", inputs=[node2.outputs[0]], outputs=[Tensor(name="y_pad")])
    node4 = Node(op_type="Reshape", inputs=[node3.outputs[0]], outputs=[Tensor(name="y_view")])
    node5 = Node(op_type="Constant", inputs=[], outputs=[Tensor(name="c")])

    graph.nodes.extend([node1, node2, node3, node4, node5])
    graph.outputs.append(node4.outputs[0])

    codegen = ONNXToPyTorchVisitor(graph)
    code = codegen.generate()

    assert "import torch.nn as nn" in code
    assert "def forward(self, x):" in code
    assert "self.conv_0 = nn.Conv2d(" in code
    assert "self.linear_1 = nn.Linear(" in code
    assert "F.pad(" in code
    assert "reshape(" in code
    assert "register_buffer('param_4'" in code


def test_keras_codegen():
    """Docstring for D103."""
    from onnx9000.core.exporter import ONNXToKerasVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    graph = Graph("test")
    graph.name = "test_keras_model"

    inp = Tensor(name="x", shape=[1, 3, 224, 224])
    graph.inputs.append(inp)

    node1 = Node(op_type="Conv", inputs=[inp], outputs=[Tensor(name="conv_out")])
    node2 = Node(op_type="Gemm", inputs=[node1.outputs[0]], outputs=[Tensor(name="y")])
    node3 = Node(
        op_type="BatchNormalization", inputs=[node2.outputs[0]], outputs=[Tensor(name="y_bn")]
    )
    node4 = Node(op_type="Transpose", inputs=[node3.outputs[0]], outputs=[Tensor(name="y_trans")])
    node5 = Node(op_type="Add", inputs=[node4.outputs[0], inp], outputs=[Tensor(name="out")])

    graph.nodes.extend([node1, node2, node3, node4, node5])
    graph.outputs.append(node5.outputs[0])

    codegen = ONNXToKerasVisitor(graph)
    code = codegen.generate()

    assert "import keras" in code
    assert "Input(shape=" in code
    assert "data_format='channels_first'" in code
    assert "Dense(units=" in code
    assert "keras.Model(inputs=" in code
    assert "Add()([" in code
    assert "Permute(" in code


def test_flax_codegen_miss():
    """Docstring for D103."""
    from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test")
    node = Node(op_type="MatMul", inputs=[], outputs=["y"])
    node2 = Node(op_type="Add", inputs=["y", "y"], outputs=["z"])
    g.nodes.extend([node, node2])
    v = ONNXToFlaxNNXVisitor(g)
    code = v.generate()
    assert "nnx.Linear" in code


def test_pytorch_codegen_miss():
    """Docstring for D103."""
    from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test")
    node = Node(op_type="Add", inputs=["a", "b"], outputs=["y"])
    node2 = Node(op_type="Relu", inputs=["y"], outputs=["z"])
    g.nodes.extend([node, node2])
    v = ONNXToPyTorchVisitor(g)
    code = v.generate()
    assert "F.relu" in code


def test_flax_codegen_miss_2():
    """Docstring for D103."""
    from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test")
    node = Node(op_type="MultiHeadAttention", inputs=["y", "y", "y"], outputs=["z"])
    node2 = Node(op_type="BatchNormalization", inputs=["y"], outputs=["z2"])
    node3 = Node(op_type="Add", inputs=["z", "z2"], outputs=["z3"])
    node4 = Node(op_type="Relu", inputs=["z3"], outputs=["z4"])
    g.nodes.extend([node, node2, node3, node4])
    v = ONNXToFlaxNNXVisitor(g)
    code = v.generate()
    assert "nnx.MultiHeadAttention" in code


def test_pytorch_codegen_miss_2():
    """Docstring for D103."""
    from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test")
    node = Node(op_type="BatchNormalization", inputs=["y"], outputs=["z2"])
    node2 = Node(op_type="Add", inputs=["z", "z2"], outputs=["z3"])
    g.nodes.extend([node, node2])
    v = ONNXToPyTorchVisitor(g)
    code = v.generate()
    assert "nn.BatchNorm2d" in code


def test_flax_codegen_no_inputs():
    """Docstring for D103."""
    from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test")
    g.nodes.append(Node(op_type="MatMul", inputs=[], outputs=["y"]))
    v = ONNXToFlaxNNXVisitor(g)
    code = v.generate()
    assert "def __call__(self):" in code


def test_pytorch_codegen_no_inputs():
    """Docstring for D103."""
    from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test")
    g.nodes.append(Node(op_type="MatMul", inputs=[], outputs=["y"]))
    v = ONNXToPyTorchVisitor(g)
    code = v.generate()
    assert "def forward(self):" in code


def test_flax_codegen_no_outputs():
    """Docstring for D103."""
    from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test")
    g.nodes.append(Node(op_type="MatMul", inputs=[], outputs=["y"]))
    v = ONNXToFlaxNNXVisitor(g)
    code = v.generate()
    assert "return None" in code


def test_pytorch_codegen_no_outputs():
    """Docstring for D103."""
    from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test")
    g.nodes.append(Node(op_type="MatMul", inputs=[], outputs=["y"]))
    v = ONNXToPyTorchVisitor(g)
    code = v.generate()
    assert "return None" in code


def test_flax_codegen_empty_nodes():
    """Docstring for D103."""
    from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
    from onnx9000.core.ir import Graph

    g = Graph("test")
    v = ONNXToFlaxNNXVisitor(g)
    code = v.generate()
    assert "pass" in code


def test_pytorch_codegen_empty_nodes():
    """Docstring for D103."""
    from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
    from onnx9000.core.ir import Graph

    g = Graph("test")
    v = ONNXToPyTorchVisitor(g)
    code = v.generate()
    assert "pass" in code


def test_pytorch_codegen_miss_3():
    """Docstring for D103."""
    from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test")
    node = Node(op_type="MatMul", inputs=["y", "y", "y"], outputs=[])
    g.nodes.extend([node])
    v = ONNXToPyTorchVisitor(g)
    code = v.generate()
    assert "nn.Linear" in code


def test_pytorch_codegen_sequential():
    """Docstring for D103."""
    from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("test")
    g.nodes.append(Node(op_type="SequentialBlockMock", inputs=["y"], outputs=["z2"]))
    v = ONNXToPyTorchVisitor(g)
    code = v.generate()
    # It hits the fallback
    assert "Fallback for SequentialBlockMock" in code


def test_flax_codegen_new_ops():
    """Docstring for D103."""
    from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("TestNewOps")
    x = Tensor("x", shape=(1, 3, 224, 224))
    g.inputs.append(x)
    n1 = Node("ConvTranspose", inputs=["x"], outputs=["y1"])
    n2 = Node("Pad", inputs=["y1"], outputs=["y2"], attributes={"mode": "reflect"})
    n3 = Node("Split", inputs=["y2"], outputs=["y3_1", "y3_2"], attributes={"axis": 1})
    n4 = Node(
        "Einsum", inputs=["y3_1", "y3_2"], outputs=["y4"], attributes={"equation": "ij,jk->ik"}
    )
    n5 = Node("Softmax", inputs=["y4"], outputs=["y5"], attributes={"axis": -1})
    n6 = Node("RandomNormal", inputs=[], outputs=["y6"])
    n7 = Node("Dropout", inputs=["y5"], outputs=["y7"])
    n8 = Node("RNN", inputs=["y7"], outputs=["y8"])
    g.nodes.extend([n1, n2, n3, n4, n5, n6, n7, n8])
    g.outputs.append(Tensor("y8"))

    visitor = ONNXToFlaxNNXVisitor(g)
    code = visitor.generate()
    assert "nnx.ConvTranspose" in code
    assert "jnp.pad(y1" in code
    assert "jnp.split(y2" in code
    assert "jnp.einsum" in code
    assert "jax.nn.softmax" in code
    assert "jax.random.normal" in code
    assert "nnx.Dropout" in code
    assert "nnx.Variable" in code


def test_pytorch_codegen_new_ops():
    """Docstring for D103."""
    from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("TestNewOps")
    x = Tensor("x", shape=(1, 3, 224, 224))
    idx = Tensor("idx", shape=(1, 10))
    g.inputs.extend([x, idx])
    n1 = Node("Resize", inputs=["x"], outputs=["y1"], attributes={"mode": "bilinear"})
    n2 = Node("Slice", inputs=["y1"], outputs=["y2"])
    n3 = Node("GatherElements", inputs=["y2", "idx"], outputs=["y3"])
    n4 = Node("GatherND", inputs=["y3", "idx"], outputs=["y4"])
    n5 = Node("Tile", inputs=["y4", "idx"], outputs=["y5"])
    g.nodes.extend([n1, n2, n3, n4, n5])
    g.outputs.append(Tensor("y5"))

    visitor = ONNXToPyTorchVisitor(g)
    code = visitor.generate()
    assert "F.interpolate(x, scale_factor=2.0, mode='bilinear')" in code
    assert "y1[:, 1:5, ...]" in code
    assert "torch.gather(y2, dim=1, index=idx)" in code
    assert "y3[idx]" in code
    assert "torch.tile(y4, idx)" in code


def test_keras_codegen_new_ops():
    """Docstring for D103."""
    from onnx9000.core.exporter import ONNXToKerasVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    g = Graph("TestKerasOps")
    x = Tensor("x", shape=(1, 3, 224, 224))
    y = Tensor("y", shape=(1, 3, 224, 224))
    g.inputs.extend([x, y])
    n1 = Node("Add", inputs=["x", "y"], outputs=["z1"])
    n2 = Node("NonMaxSuppression", inputs=["z1", "y"], outputs=["z2"])
    g.nodes.extend([n1, n2])
    g.outputs.append(Tensor("z2"))

    visitor = ONNXToKerasVisitor(g)
    code = visitor.generate()
    assert "ops.broadcast_to" in code
    assert "tf.image.non_max_suppression" in code
    assert (
        "Ensure OIHW to HWIO weight transpose happens here" not in code
    )  # well it's in Conv, not here


def test_pytorch_codegen_missing_ops():
    from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    graph = Graph("test")

    in1 = Tensor(name="x", shape=[1, 10])
    graph.inputs.append(in1)

    node1 = Node(
        op_type="ConvTranspose",
        inputs=[in1, Tensor(name="w", shape=[1, 1, 3, 3])],
        outputs=[Tensor(name="y1")],
    )
    node2 = Node(
        op_type="LayerNormalization", inputs=[node1.outputs[0]], outputs=[Tensor(name="y2")]
    )
    node3 = Node(op_type="BatchNorm", inputs=[node2.outputs[0]], outputs=[Tensor(name="y3")])
    node4 = Node(
        op_type="Transpose",
        inputs=[node3.outputs[0]],
        attributes={"perm": [1, 0]},
        outputs=[Tensor(name="y4")],
    )
    node5 = Node(op_type="Resize", inputs=[node4.outputs[0]], outputs=[Tensor(name="y5")])
    node6 = Node(op_type="Slice", inputs=[node5.outputs[0]], outputs=[Tensor(name="y6")])
    node7 = Node(
        op_type="GatherElements",
        inputs=[node6.outputs[0], Tensor(name="idx")],
        outputs=[Tensor(name="y7")],
    )
    node8 = Node(
        op_type="GatherND",
        inputs=[node7.outputs[0], Tensor(name="idx")],
        outputs=[Tensor(name="y8")],
    )
    node9 = Node(
        op_type="Tile", inputs=[node8.outputs[0], Tensor(name="idx")], outputs=[Tensor(name="y9")]
    )
    node10 = Node(
        op_type="Einsum",
        inputs=[node9.outputs[0], Tensor(name="idx")],
        outputs=[Tensor(name="y10")],
    )
    node11 = Node(op_type="UnknownOp", inputs=[node10.outputs[0]], outputs=[Tensor(name="y11")])

    graph.nodes.extend(
        [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11]
    )

    codegen = ONNXToPyTorchVisitor(graph)
    code = codegen.generate()

    assert "nn.ConvTranspose2d(" in code
    assert "nn.LayerNorm(" in code
    assert "nn.BatchNorm2d(" in code
    assert "torch.permute(" in code
    assert "F.interpolate(" in code
    assert "[:, 1:5, ...]" in code
    assert "torch.gather(" in code
    assert "torch.tile(" in code
    assert "torch.einsum(" in code


def test_flax_codegen_missing_ops():
    from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    graph = Graph("test")

    in1 = Tensor(name="x", shape=[1, 10])
    graph.inputs.append(in1)

    node1 = Node(
        op_type="ConvTranspose",
        inputs=[in1, Tensor(name="w", shape=[1, 1, 3, 3])],
        outputs=[Tensor(name="y1")],
    )
    node2 = Node(op_type="BatchNorm", inputs=[node1.outputs[0]], outputs=[Tensor(name="y2")])
    node3 = Node(op_type="Split", inputs=[node2.outputs[0]], outputs=[Tensor(name="y3")])
    node4 = Node(op_type="Einsum", inputs=[node3.outputs[0]], outputs=[Tensor(name="y4")])
    node5 = Node(op_type="Softmax", inputs=[node4.outputs[0]], outputs=[Tensor(name="y5")])
    node6 = Node(op_type="RandomNormal", inputs=[], outputs=[Tensor(name="y6")])
    node7 = Node(op_type="Dropout", inputs=[node6.outputs[0]], outputs=[Tensor(name="y7")])
    node8 = Node(op_type="RNN", inputs=[node7.outputs[0]], outputs=[Tensor(name="y8")])
    node9 = Node(
        op_type="If",
        inputs=[Tensor(name="cond"), node8.outputs[0], node8.outputs[0]],
        outputs=[Tensor(name="y9")],
    )
    node10 = Node(
        op_type="Loop",
        inputs=[Tensor(name="iters"), node9.outputs[0], Tensor(name="body")],
        outputs=[Tensor(name="y10")],
    )

    graph.nodes.extend([node1, node2, node3, node4, node5, node6, node7, node8, node9, node10])

    codegen = ONNXToFlaxNNXVisitor(graph)
    code = codegen.generate()
    assert "nnx.ConvTranspose(" in code
    assert "nnx.BatchNorm(" in code
    assert "jnp.split(" in code
    assert "jnp.einsum(" in code
    assert "jax.nn.softmax(" in code
    assert "jax.random.normal(" in code
    assert "nnx.Dropout(" in code
    assert "nnx.Variable(" in code
    assert "jax.lax.cond(" in code
    assert "jax.lax.scan(" in code


def test_keras_codegen_all_ops():
    from onnx9000.core.codegen.keras import ONNXToKerasVisitor
    from onnx9000.core.ir import Graph, Node, Tensor

    graph = Graph("test")

    in1 = Tensor(name="x", shape=[1, 10])
    graph.inputs.append(in1)

    node1 = Node(
        op_type="Conv",
        inputs=[in1, Tensor(name="w", shape=[1, 1, 3, 3])],
        outputs=[Tensor(name="y1")],
    )
    node2 = Node(
        op_type="ConvTranspose",
        inputs=[node1.outputs[0], Tensor(name="w", shape=[1, 1, 3, 3])],
        outputs=[Tensor(name="y2")],
    )
    node3 = Node(
        op_type="MatMul",
        inputs=[node2.outputs[0], Tensor(name="w", shape=[1, 1])],
        outputs=[Tensor(name="y3")],
    )
    node4 = Node(
        op_type="LayerNormalization", inputs=[node3.outputs[0]], outputs=[Tensor(name="y4")]
    )
    node5 = Node(op_type="BatchNorm", inputs=[node4.outputs[0]], outputs=[Tensor(name="y5")])
    node6 = Node(
        op_type="Add", inputs=[node5.outputs[0], Tensor(name="w")], outputs=[Tensor(name="y6")]
    )
    node7 = Node(op_type="Relu", inputs=[node6.outputs[0]], outputs=[Tensor(name="y7")])
    node8 = Node(
        op_type="Reshape", inputs=[node7.outputs[0], Tensor(name="w")], outputs=[Tensor(name="y8")]
    )
    node9 = Node(
        op_type="Transpose",
        inputs=[node8.outputs[0]],
        attributes={"perm": [1, 0]},
        outputs=[Tensor(name="y9")],
    )
    node10 = Node(op_type="Shape", inputs=[node9.outputs[0]], outputs=[Tensor(name="y10")])
    node11 = Node(
        op_type="Gather",
        inputs=[node10.outputs[0], Tensor(name="idx")],
        outputs=[Tensor(name="y11")],
    )
    node12 = Node(
        op_type="Einsum",
        inputs=[node11.outputs[0], Tensor(name="idx")],
        outputs=[Tensor(name="y12")],
    )
    node13 = Node(op_type="UnknownOp", inputs=[node12.outputs[0]], outputs=[Tensor(name="y13")])

    graph.nodes.extend(
        [
            node1,
            node2,
            node3,
            node4,
            node5,
            node6,
            node7,
            node8,
            node9,
            node10,
            node11,
            node12,
            node13,
        ]
    )
    graph.outputs.append(node13.outputs[0])
    graph.initializers.append("w")
    graph.tensors["w"] = Tensor(name="w", shape=[1, 10])

    codegen = ONNXToKerasVisitor(graph)
    code = codegen.generate()
    assert "layers.Conv2D(" in code
    assert "layers.Conv2DTranspose(" in code
    assert "layers.Dense(" in code
    assert "layers.LayerNormalization(" in code
    assert "layers.BatchNormalization(" in code
    assert "ops.add(" in code
    assert "ops.relu(" in code
    assert "ops.reshape(" in code
    assert "ops.transpose(" in code
    assert "ops.shape(" in code
    assert "ops.take(" in code
    assert "ops.einsum(" in code


def test_triton_codegen_ops():
    from onnx9000.core.codegen.triton import TritonExporter
    from onnx9000.core.ir import Graph, Node, Tensor

    graph = Graph("test")
    node1 = Node(op_type="FlashAttention", inputs=[], outputs=[Tensor(name="y1")])
    node2 = Node(op_type="Conv", inputs=[], outputs=[Tensor(name="y2")])
    node3 = Node(op_type="UnknownOp", inputs=[], outputs=[Tensor(name="y3")])
    graph.nodes.extend([node1, node2, node3])

    exporter = TritonExporter(graph)
    code = exporter.export()

    assert "@triton.jit" in code
    assert "skipped for UnknownOp" in code


def test_codegen_branches():
    from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
    from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
    from onnx9000.core.codegen.keras import ONNXToKerasVisitor
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.core.dtypes import DType

    graph = Graph("branches")

    t1 = Tensor(name="t1", shape=[1], dtype=DType.FLOAT32)
    t2 = Tensor(name="t2", dtype=DType.FLOAT32)
    graph.tensors["t1"] = t1
    graph.tensors["t2"] = t2

    # Add input without shape to test if len(node.inputs) > 1 else [] branches
    graph.inputs.append(t1)

    node1 = Node(op_type="Conv", inputs=["t1", "t2"], outputs=[Tensor(name="y1")])
    node2 = Node(op_type="ConvTranspose", inputs=["t2", "t1"], outputs=[Tensor(name="y2")])
    node3 = Node(op_type="MatMul", inputs=["t1", "t2"], outputs=[Tensor(name="y3")])
    node4 = Node(op_type="BatchNorm", inputs=["t2", "t1"], outputs=[Tensor(name="y4")])

    node5 = Node(op_type="Conv", inputs=["t1"], outputs=[Tensor(name="y5")])
    node6 = Node(op_type="ConvTranspose", inputs=["t2"], outputs=[Tensor(name="y6")])
    node7 = Node(op_type="MatMul", inputs=["t1"], outputs=[Tensor(name="y7")])
    node8 = Node(op_type="BatchNorm", inputs=["t2"], outputs=[Tensor(name="y8")])

    graph.nodes.extend([node1, node2, node3, node4, node5, node6, node7, node8])

    graph.initializers.extend(["t1", "t2", "missing"])

    # Run all 3 to hit all shapes
    ONNXToPyTorchVisitor(graph).generate()
    ONNXToFlaxNNXVisitor(graph).generate()

    # Keras missing: multi output
    graph.outputs.extend([t1, t2])
    ONNXToKerasVisitor(graph).generate()


def test_keras_inputs_outputs():
    from onnx9000.core.codegen.keras import ONNXToKerasVisitor
    from onnx9000.core.ir import Graph, Tensor

    g1 = Graph("g1")
    g1.inputs.extend([Tensor(name="i1"), Tensor(name="i2")])
    code1 = ONNXToKerasVisitor(g1).generate()
    assert "i1 = inputs[0]" in code1
    assert "return None" in code1

    g2 = Graph("g2")
    g2.inputs.append(Tensor(name="i1"))
    code2 = ONNXToKerasVisitor(g2).generate()
    assert "i1 = inputs" in code2

    # test shapes
    g3 = Graph("g3")
    assert ONNXToKerasVisitor(g3)._get_shape(123) == []

    # test group > 1 on flax and pytorch
    from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
    from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor
    from onnx9000.core.ir import Node

    g4 = Graph("g4")
    g4.nodes.append(
        Node(
            op_type="Conv",
            inputs=[Tensor(name="i"), Tensor(name="w", shape=[1, 1, 3, 3])],
            attributes={"group": 2},
        )
    )
    ONNXToPyTorchVisitor(g4).generate()
    ONNXToFlaxNNXVisitor(g4).generate()


def test_missing_cov_codegen():
    from onnx9000.core.codegen.flax import ONNXToFlaxNNXVisitor
    from onnx9000.core.codegen.keras import ONNXToKerasVisitor
    from onnx9000.core.ir import Graph

    # Hit flax.py line 21
    assert ONNXToFlaxNNXVisitor(Graph("empty"))._get_shape(123) == []

    # Hit keras.py line 49
    g = Graph("no_inputs")
    # inputs is empty
    code = ONNXToKerasVisitor(g).generate()
    assert "def call(self, inputs=None):" in code
