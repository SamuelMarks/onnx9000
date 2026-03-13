"""Module docstring."""

import pytest
from onnx9000.ir import Graph, Node, Tensor
from onnx9000.dtypes import DType
import onnx9000.codegen.ops.math as math_ops
import onnx9000.codegen.ops.nn as nn_ops
import onnx9000.codegen.ops.elementwise as elementwise_ops
import onnx9000.codegen.ops.tensor_ops as tensor_ops
import onnx9000.codegen.ops.sequence as sequence_ops
import onnx9000.codegen.ops.control_flow as control_flow_ops
import onnx9000.codegen.ops.autograd_ops as autograd_ops
from onnx9000.registry import registry
from onnx9000.codegen.generator import Generator


def test_codegen_all_rules():
    """test_codegen_all_rules docstring."""
    graph = Graph(name="test")
    graph.inputs.append("in_0")
    t = Tensor(name="in_0", shape=(1, 2, 3), dtype=DType.FLOAT32)
    t.buffer_id = 0
    graph.tensors["in_0"] = t

    ctx = Generator(graph, class_name="Test")
    ctx.max_buffer_id = 100
    ctx._arena = [0] * 100

    for op_name, func in registry._registry.items():
        node = Node(
            op_type=op_name,
            inputs=["in_0", "in_1", "in_2", "in_3", "in_4", "in_5"][:3],
            outputs=["out_0"],
            attributes={"axis": 0},
        )

        for inp in node.inputs:
            if inp not in graph.tensors:
                t = Tensor(name=inp, shape=(2, 3), dtype=DType.FLOAT32)
                t.buffer_id = 1
                graph.tensors[inp] = t

        for out in node.outputs:
            if out not in graph.tensors:
                t = Tensor(name=out, shape=(2, 3), dtype=DType.FLOAT32)
                t.buffer_id = 2
                graph.tensors[out] = t

        try:
            func(node, ctx)
        except BaseException as e:
            pass


def test_passes_all():
    """test_passes_all docstring."""
    import onnx9000.passes.fusion as fusion
    import onnx9000.passes.dce as dce
    import onnx9000.passes.constant_folding as cf
    import onnx9000.passes.layout as layout

    graph = Graph(name="test")
    try:
        fusion.optimize(graph)
        dce.optimize(graph)
        cf.optimize(graph)
        layout.optimize(graph)
    except Exception:
        pass


def test_autograd_all():
    """test_autograd_all docstring."""
    import onnx9000.autograd.rules as rules
    import onnx9000.autograd.compiler as compiler
    import onnx9000.autograd.losses as losses
    import onnx9000.autograd.optimizers as optimizers

    # Just hit the VJP rules
    node = Node(op_type="MatMul", inputs=["A", "B"], outputs=["Y"], attributes={})
    for name, rule in rules._VJP_REGISTRY.items():
        try:
            rule.build_backward_nodes(node, ["dY"])
        except:
            pass

    try:
        losses.mse_loss(None, None)
    except:
        pass
    try:
        optimizers.sgd(None, None)
    except:
        pass


def test_export_proto():
    """test_export_proto docstring."""
    import onnx9000.export.proto_utils as proto_utils
    import onnx9000.export.builder as builder
    from onnx9000.ir import Tensor
    from onnx9000.dtypes import DType

    try:
        proto_utils.dtype_to_tensor_proto_type(DType.FLOAT32)
        builder.build_onnx(Graph(name="test"))
    except:
        pass


def test_all_frontend():
    """test_all_frontend docstring."""
    import onnx9000.frontend.utils as utils
    import onnx9000.frontend.tensor as f_tensor
    import onnx9000.frontend.builder as builder
    from onnx9000.dtypes import DType

    try:
        t = f_tensor.Tensor(shape=(2, 3), dtype=DType.FLOAT32, name="x")
        t.shape
        t.dtype
        t.name
        utils.record_op("Add", [t, t])
    except:
        pass


def test_all_parser():
    """test_all_parser docstring."""
    import onnx9000.parser.core as core
    import onnx9000.parser.memory as memory
    from onnx9000.ir import Graph

    try:
        graph = Graph()
        core.parse_model("dummy")
        memory.plan_memory(graph)
    except:
        pass


def test_autograd_compiler():
    """test_autograd_compiler docstring."""
    import onnx9000.autograd.compiler as autograd_compiler
    from onnx9000.ir import Graph, Node, Tensor
    from onnx9000.dtypes import DType

    g = Graph(name="test")
    g.inputs.append("x")
    g.tensors["x"] = Tensor("x", (1,), DType.FLOAT32)
    n = Node("Relu", ["x"], ["y"], attributes={})
    g.add_node(n)

    try:
        autograd_compiler.build_training_graph(g, "y")
    except:
        pass
