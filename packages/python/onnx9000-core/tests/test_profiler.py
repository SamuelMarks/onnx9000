"""Tests the profiler module functionality."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Constant, DynamicDim, Graph, Node, Tensor
from onnx9000.core.profiler import ProfilerResult, profile


def test_profiler_macs_flops_basic():
    """Tests the profiler macs flops basic functionality."""
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("B", shape=(20, 30), dtype=DType.FLOAT32))
    g.inputs.extend(["A", "B"])

    n1 = Node("MatMul", inputs=["A", "B"], outputs=["Y"])
    g.add_node(n1)

    res = g.profile()

    # MatMul MACs = 10 * 30 * 20 = 6000
    # FLOPs = 12000
    assert res.total_macs == 6000
    assert res.total_flops == 12000
    assert len(res.node_profiles) == 1


def test_profiler_conv():
    """Tests the profiler conv functionality."""
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(1, 3, 224, 224), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("W", shape=(64, 3, 7, 7), dtype=DType.FLOAT32))
    g.inputs.extend(["X", "W"])

    n = Node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        attributes={
            "kernel_shape": Attribute("kernel_shape", value=[7, 7]),
            "strides": Attribute("strides", value=[2, 2]),
            "pads": Attribute("pads", value=[3, 3, 3, 3]),
        },
    )
    g.add_node(n)
    res = g.profile()

    # Out volume = 1 * 64 * 112 * 112 = 802816
    # k_vol = 7 * 7 = 49
    # in_c = 3
    # MACs = 802816 * 49 * 3 = 118013952
    assert res.total_macs == 118013952
    assert res.total_flops == 118013952 * 2


def test_profiler_symbolic():
    """Tests the profiler symbolic functionality."""
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(DynamicDim("B"), 20), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("W", shape=(20, 30), dtype=DType.FLOAT32))
    g.inputs.extend(["A", "W"])

    n1 = Node("MatMul", inputs=["A", "W"], outputs=["Y"])
    g.add_node(n1)

    # Evaluate with override
    res = g.profile(dynamic_overrides={"B": 10})
    assert res.total_macs == 6000

    # Evaluate without override
    res2 = g.profile()
    assert (
        res2.total_macs == "(B * 30 * 20)"
        or res2.total_macs == "(30 * B * 20)"
        or "B" in str(res2.total_macs)
    )


def test_profiler_memory_params():
    """Tests the profiler memory params functionality."""
    g = Graph("test")
    w = Constant("W", shape=(10, 10), dtype=DType.FLOAT32, values=b"\0" * 400)
    g.add_tensor(w)
    res = g.profile()
    assert res.total_params == 100
    assert res.total_memory_bytes == 400


def test_profiler_bandwidth_and_precisions():
    """Tests the profiler bandwidth and precisions functionality."""
    g = Graph("test")
    g.add_tensor(Tensor("X", shape=(10, 20), dtype=DType.FLOAT16))
    g.add_tensor(Tensor("W", shape=(20, 30), dtype=DType.FLOAT16))
    g.inputs.extend(["X", "W"])

    n1 = Node("MatMul", inputs=["X", "W"], outputs=["Y"], name="n1")
    g.add_node(n1)

    # Reshape is memory bound
    shape_t = Tensor("shape", shape=(2,), dtype=DType.INT64)
    shape_t.values = [10, 30]
    g.add_tensor(shape_t)
    n2 = Node("Reshape", inputs=["Y", "shape"], outputs=["Z"], name="n2")
    g.add_node(n2)

    res = g.profile()

    # MatMul MACs = 10 * 30 * 20 = 6000
    assert res.fp16_macs == 6000
    assert res.fp32_macs == 0

    # Reshape mem bandwidth
    # Input Y is (10, 30) FLOAT16 -> 10*30*2 = 600 bytes
    # Output Z is (10, 30) FLOAT16 -> 600 bytes
    # total bw = 1200 bytes
    reshape_node = next(x for x in res.node_profiles if x["name"] == "n2")
    assert reshape_node["mem_bandwidth_bytes"] == 1200
    assert reshape_node["flops"] == 0

    cum_flops = res.get_cumulative_flops_up_to("n1")
    assert cum_flops == 12000


def test_profiler_latency_and_suggestions():
    """Tests the profiler latency and suggestions functionality."""
    g = Graph("test")
    g.add_tensor(Tensor("A", shape=(10, 20), dtype=DType.FLOAT64, is_initializer=True))
    g.add_tensor(Tensor("W", shape=(20, 30), dtype=DType.INT64, is_initializer=True))
    g.inputs.extend(["A", "W"])

    n1 = Node("MatMul", inputs=["A", "huge"], outputs=["Y"], name="large_node")
    g.add_node(n1)

    # Simulate a huge tensor
    large_t = Constant("huge", shape=(10000, 10000), dtype=DType.FLOAT32, values=b"\0" * 400000000)
    g.add_tensor(large_t)

    res = g.profile()
    res.generate_suggestions()

    assert res.float64_count == 1
    assert res.int64_count == 1

    found_float64 = False
    found_int64 = False
    found_webgpu = False

    for s in res.suggestions:
        if "Float64" in s:
            found_float64 = True
        if "Int64" in s:
            found_int64 = True
        if "exceeds WebGPU" in s:
            found_webgpu = True

    assert found_float64
    assert found_int64
    assert found_webgpu

    latency = res.estimate_latency(hardware_tops=1.0, hardware_bw_gbps=10.0)
    assert "total_estimated_latency_ms" in latency
    assert latency["bottleneck"] == "Memory Bound"
