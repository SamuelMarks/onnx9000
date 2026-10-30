"""Comprehensive unit tests for simplifier optimization passes (broadcast, debug, flattening, layout, memory, partitioning, quantization, versioning)."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, DynamicDim, Graph, Node, Tensor
from onnx9000.optimizer.simplifier.passes.base import GraphPass
from onnx9000.optimizer.simplifier.passes.broadcast import optimize_broadcasting
from onnx9000.optimizer.simplifier.passes.debug import inject_probes
from onnx9000.optimizer.simplifier.passes.flattening import flatten_subgraphs
from onnx9000.optimizer.simplifier.passes.layout import (
    transform_nchw_to_nhwc,
    transform_nhwc_to_nchw,
)
from onnx9000.optimizer.simplifier.passes.memory_planning import (
    estimate_memory_consumption,
    plan_tensor_lifecycles,
)
from onnx9000.optimizer.simplifier.passes.partitioning import partition_for_multi_device
from onnx9000.optimizer.simplifier.passes.quantization import (
    convert_to_int8,
    insert_qat_nodes,
)
from onnx9000.optimizer.simplifier.passes.versioning import (
    apply_opset_fallbacks,
    enforce_opset_18,
)
from onnx9000.optimizer.simplifier.passes.webgpu import (
    optimize_for_webgpu,
    polyfill_webgpu_unsupported,
)


class DummyPass(GraphPass):
    """Concrete GraphPass subclass for testing base pass invocation."""

    def run(self, graph: Graph) -> bool:
        """Invoke base run and return True."""
        super().run(graph)
        return True


def test_base_pass() -> None:
    """Verify GraphPass base class execution."""
    p = DummyPass()
    assert p.run(Graph("base")) is True


def test_empty_graph_passes() -> None:
    """Verify all passes return False or empty mappings on an empty Graph."""
    g = Graph("test")
    assert optimize_broadcasting(g) is False
    assert inject_probes(g) is False
    assert flatten_subgraphs(g) is False
    assert transform_nchw_to_nhwc(g) is False
    assert transform_nhwc_to_nchw(g) is False
    assert estimate_memory_consumption(g) == {}
    assert plan_tensor_lifecycles(g) == {}
    assert partition_for_multi_device(g) == {"device_0": g}
    assert insert_qat_nodes(g) is False
    assert convert_to_int8(g) is False
    assert apply_opset_fallbacks(g) is False
    assert enforce_opset_18(g) is False
    assert polyfill_webgpu_unsupported(g) is False
    assert optimize_for_webgpu(g) is False


def test_broadcast_pass_active() -> None:
    """Verify optimize_broadcasting bypasses Expand nodes feeding binary ops."""
    g_no_exp = Graph("no_exp")
    g_no_exp.add_node(Node("Relu", ["a"], ["b"]))
    assert optimize_broadcasting(g_no_exp) is False

    g = Graph("bcast")
    g.add_node(Node("Expand", ["x", "shape"], ["x_exp"]))
    g.add_node(Node("Add", ["x_exp", "y"], ["z"]))
    g.add_node(Node("Relu", ["z"], ["out"]))
    assert optimize_broadcasting(g) is True
    assert g.nodes[1].inputs == ["x", "y"]


def test_debug_inject_probes_active() -> None:
    """Verify inject_probes exposes intermediate outputs without duplicating existing outputs."""
    g = Graph("debug")
    g.outputs.append("final")
    g.add_node(Node("Relu", ["x"], ["mid"]))
    g.add_node(Node("Abs", ["mid"], ["final"]))
    assert inject_probes(g) is True
    assert len(g.outputs) == 2
    assert inject_probes(g) is False


def test_flatten_subgraphs_active() -> None:
    """Verify flatten_subgraphs inlines nested Graph attributes."""
    sub = Graph("sub")
    sub.add_node(Node("Relu", ["x"], ["y"]))

    g = Graph("parent")
    g.add_node(
        Node(
            "If",
            ["cond"],
            ["y"],
            attributes={"then_branch": Attribute("then_branch", "GRAPH", sub)},
        )
    )
    g.add_node(Node("Identity", ["y"], ["z"], attributes={"axis": Attribute("axis", "INT", 0)}))
    assert flatten_subgraphs(g) is True
    assert g.nodes[0].op_type == "Relu"
    assert flatten_subgraphs(g) is False


def test_layout_transforms_active() -> None:
    """Verify NCHW <-> NHWC layout annotation transformations."""
    g = Graph("layout")
    g.add_node(Node("Conv", ["x", "w"], ["y"]))
    g.add_node(Node("Relu", ["y"], ["z"]))

    assert transform_nchw_to_nhwc(g) is True
    assert g.nodes[0].attributes["data_format"].value == "NHWC"
    assert transform_nchw_to_nhwc(g) is False

    assert transform_nhwc_to_nchw(g) is True
    assert g.nodes[0].attributes["data_format"].value == "NCHW"
    assert transform_nhwc_to_nchw(g) is False


def test_memory_planning_and_partitioning_active() -> None:
    """Verify estimate_memory_consumption, plan_tensor_lifecycles, and multi-device partitioning."""
    g = Graph("mem")
    g.add_tensor(Tensor("A", (2, DynamicDim(4), DynamicDim("N"), -1), DType.FLOAT32))
    g.add_tensor(Tensor("B", (), "float32"))
    g.add_node(Node("Add", ["A", "B"], ["C"]))
    g.add_node(Node("Relu", ["C"], ["C"]))
    g.add_node(Node("Abs", ["C"], ["D"]))

    mem = estimate_memory_consumption(g)
    assert mem["A"] == 32
    assert mem["B"] == 4

    lifespans = plan_tensor_lifecycles(g)
    assert lifespans["A"] == (0, 0)
    assert lifespans["C"] == (0, 2)
    assert lifespans["D"] == (2, 2)

    parts = partition_for_multi_device(g, num_devices=2)
    assert len(parts) == 2
    assert len(parts["device_0"].nodes) == 2
    assert len(parts["device_1"].nodes) == 1


def test_quantization_and_versioning_active() -> None:
    """Verify insert_qat_nodes, convert_to_int8, apply_opset_fallbacks, and enforce_opset_18."""
    g = Graph("quant_ver")
    g.add_node(Node("SpatialBN", ["x"], ["y"]))
    g.add_node(Node("Conv", ["y", "w"], ["z"]))

    assert apply_opset_fallbacks(g) is True
    assert g.nodes[0].op_type == "BatchNormalization"
    assert apply_opset_fallbacks(g) is False

    assert insert_qat_nodes(g) is True
    assert "qat_enabled" in g.nodes[1].attributes
    assert insert_qat_nodes(g) is False

    assert convert_to_int8(g) is True
    assert g.nodes[1].op_type == "QLinearConv"
    assert convert_to_int8(g) is False

    assert enforce_opset_18(g) is True
    assert g.opset_imports[""] == 18
    assert enforce_opset_18(g) is False
