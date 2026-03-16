from onnx9000.core.ir import Graph
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
from onnx9000.optimizer.simplifier.passes.quantization import convert_to_int8, insert_qat_nodes
from onnx9000.optimizer.simplifier.passes.versioning import apply_opset_fallbacks, enforce_opset_18
from onnx9000.optimizer.simplifier.passes.webgpu import (
    optimize_for_webgpu,
    polyfill_webgpu_unsupported,
)


class DummyPass(GraphPass):
    def run(self, graph: Graph) -> bool:
        super().run(graph)
        return True


def test_base_pass() -> None:
    p = DummyPass()
    assert p.run(None) is True


def test_stub_passes() -> None:
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
