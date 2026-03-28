"""Test pipeline optimizer."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimizer.hardware.pipeline import PipelineOptimizer


def create_mock_pipeline_graph():
    """Test the create_mock_pipeline_graph functionality."""
    g = Graph("pipeline_graph")
    g.inputs = ["in1", "in2"]
    g.outputs = ["out1"]
    n1 = Node("Add", ["in1", "in2"], ["temp1"], {}, "add1")
    n2 = Node("Mul", ["temp1", "in2"], ["temp2"], {}, "mul1")
    n3 = Node("Relu", ["temp2"], ["out1"], {}, "relu1")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.initializers = ["temp1"]
    t1 = Tensor("in1", (1,), DType.FLOAT32)
    t2 = Tensor("in2", (1,), DType.FLOAT32)
    g.add_tensor(t1)
    g.add_tensor(t2)
    return g


def test_identify_independent_paths() -> None:
    """Tests the test_identify_independent_paths functionality."""
    g = create_mock_pipeline_graph()
    paths = PipelineOptimizer.identify_independent_paths(g)
    assert len(paths) == 3


def test_partition_for_webworkers() -> None:
    """Tests the test_partition_for_webworkers functionality."""
    g = create_mock_pipeline_graph()
    partitions = PipelineOptimizer.partition_for_webworkers(g, num_workers=2)
    assert len(partitions) == 2
    assert len(partitions[0].nodes) + len(partitions[1].nodes) == 3


def test_communication_via_shared_array_buffer() -> None:
    """Tests the test_communication_via_shared_array_buffer functionality."""
    g = create_mock_pipeline_graph()
    partitions = PipelineOptimizer.partition_for_webworkers(g, num_workers=2)
    config = PipelineOptimizer.communication_via_shared_array_buffer(partitions)
    assert config["partitions"] == 2


def test_schedule_critical_path() -> None:
    """Tests the test_schedule_critical_path functionality."""
    g = create_mock_pipeline_graph()
    nodes = PipelineOptimizer.schedule_critical_path(g)
    assert len(nodes) == 3


def test_memory_pooling_hints() -> None:
    """Tests the test_memory_pooling_hints functionality."""
    g = create_mock_pipeline_graph()
    g2 = PipelineOptimizer.memory_pooling_hints(g)
    assert "pool_id" in g2.nodes[0].attributes


def test_inject_alloc_free() -> None:
    """Tests the test_inject_alloc_free functionality."""
    g = create_mock_pipeline_graph()
    g2 = PipelineOptimizer.inject_alloc_free(g)
    assert len(g2.nodes) > len(g.nodes)
    has_alloc = any(n.op_type == "Alloc" for n in g2.nodes)
    has_free = any(n.op_type == "Free" for n in g2.nodes)
    assert has_alloc
    assert has_free


def test_check_alloc_free_schema() -> None:
    """Tests the test_check_alloc_free_schema functionality."""
    assert PipelineOptimizer.check_alloc_free_schema() is True


def test_cpu_vs_gpu_heuristic() -> None:
    """Tests the test_cpu_vs_gpu_heuristic functionality."""
    assert PipelineOptimizer.cpu_vs_gpu_heuristic(2 * 1024 * 1024, "Conv") == "WebGPU"
    assert PipelineOptimizer.cpu_vs_gpu_heuristic(500 * 1024, "Conv") == "WASM"
    assert PipelineOptimizer.cpu_vs_gpu_heuristic(5 * 1024 * 1024, "Add") == "WebGPU"


def test_device_placement_pass() -> None:
    """Tests the test_device_placement_pass functionality."""
    g = create_mock_pipeline_graph()
    g2 = PipelineOptimizer.device_placement_pass(g)
    assert "device" in g2.nodes[0].attributes


def test_merge_tiny_ops() -> None:
    """Tests the test_merge_tiny_ops functionality."""
    g = create_mock_pipeline_graph()
    g2 = PipelineOptimizer.merge_tiny_ops(g)
    assert len(g2.nodes) < len(g.nodes)


def test_precalculate_static_shapes() -> None:
    """Tests the test_precalculate_static_shapes functionality."""
    g = create_mock_pipeline_graph()
    t = Tensor("in1", ("dynamic",), DType.FLOAT32)
    g.add_tensor(t)
    g2 = PipelineOptimizer.precalculate_static_shapes(g, {"in1": (1, 10)})
    assert g2.tensors["in1"].shape == (1, 10)


def test_generate_static_graph() -> None:
    """Tests the test_generate_static_graph functionality."""
    g = create_mock_pipeline_graph()
    g2 = PipelineOptimizer.generate_static_graph(g)
    assert len(g2.nodes) == len(g.nodes)


def test_auto_tuner() -> None:
    """Tests the test_auto_tuner functionality."""
    g = create_mock_pipeline_graph()
    res = PipelineOptimizer.auto_tuner(g)
    assert "best_latency_ms" in res


def test_genetic_algorithm_auto_tuner() -> None:
    """Tests the test_genetic_algorithm_auto_tuner functionality."""
    g = create_mock_pipeline_graph()
    res = PipelineOptimizer.genetic_algorithm_auto_tuner(g)
    assert "best_fitness" in res


def test_integrate_auto_tuner_js() -> None:
    """Tests the test_integrate_auto_tuner_js functionality."""
    js = PipelineOptimizer.integrate_auto_tuner_js()
    assert isinstance(js, str)


def test_auto_tuner_fallback() -> None:
    """Tests the test_auto_tuner_fallback functionality."""
    fb = PipelineOptimizer.auto_tuner_fallback()
    assert isinstance(fb, str)


def test_device_placement_pass_real() -> None:
    """Tests the test_device_placement_pass_real functionality."""
    g = Graph("mock")
    t1 = Tensor("in1", (1, 1024, 1024), DType.FLOAT32)
    g.add_tensor(t1)
    n = Node("Conv", ["in1"], ["out"], {}, "c1")
    g.add_node(n)
    g2 = PipelineOptimizer.device_placement_pass(g)
    assert g2.nodes[0].attributes["device"] == "WebGPU"


def test_merge_tiny_ops_real() -> None:
    """Tests the test_merge_tiny_ops_real functionality."""
    g = Graph("mock")
    n1 = Node("Add", ["in1", "in2"], ["t1"], {}, "add1")
    n2 = Node("Relu", ["t1"], ["out"], {}, "relu1")
    g.add_node(n1)
    g.add_node(n2)
    g2 = PipelineOptimizer.merge_tiny_ops(g)
    has_fused = any(node.op_type == "FusedElementwise" for node in g2.nodes)
    assert has_fused


def test_generate_static_graph_real() -> None:
    """Tests the test_generate_static_graph_real functionality."""
    g = Graph("mock")
    n1 = Node("Add", ["in1", "in2"], ["out"], {"dynamic": True}, "add1")
    g.add_node(n1)
    g2 = PipelineOptimizer.generate_static_graph(g)
    assert "dynamic" not in g2.nodes[0].attributes
    assert "static" in g2.nodes[0].attributes


def test_genetic_algorithm_auto_tuner_empty() -> None:
    """Tests the test_genetic_algorithm_auto_tuner_empty functionality."""
    g = Graph("mock")
    res = PipelineOptimizer.genetic_algorithm_auto_tuner(g)
    assert res["best_fitness"] == 1.0
