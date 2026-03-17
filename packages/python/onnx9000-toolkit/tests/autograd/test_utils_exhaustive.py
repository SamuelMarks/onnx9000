import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.utils import (
    GradientProto,
    generate_gradient_proto,
    calculate_gradient_payload_size,
    compress_gradients_int8,
    compile_multi_replica_graph,
    embed_distributed_identifiers,
    add_synchronous_barrier,
    calculate_communication_bounds,
    flatten_gradients,
    reverse_topological_sort,
)


def test_gradient_proto():
    proto = GradientProto("g1", "w1", "grad_w1")
    assert proto.name == "g1"
    assert proto.weight_name == "w1"
    assert proto.gradient_name == "grad_w1"


def test_generate_gradient_proto():
    fwd_graph = Graph(name="fwd")
    fwd_graph.initializers.append("w1")
    fwd_graph.initializers.append("w2")

    bwd_graph = Graph(name="bwd")
    bwd_graph.outputs.extend(["grad_w1", "grad_w2", "grad_w3", "loss"])

    protos = generate_gradient_proto(fwd_graph, bwd_graph)
    assert set([p.weight_name for p in protos]) == {"w1", "w2"}


def test_calculate_gradient_payload_size():
    graph = Graph(name="g")

    # Need to mock the shapes and types
    graph.add_tensor(Tensor(name="grad_w1", shape=(10, 10), dtype="float32"))  # 400
    graph.add_tensor(Tensor(name="grad_w2", shape=(5,), dtype="float16"))  # 10
    graph.add_tensor(Tensor(name="grad_w3", shape=(2, 2), dtype="int8"))  # 4
    graph.add_tensor(Tensor(name="grad_w4", shape=(2,), dtype="int32"))  # 8
    graph.add_tensor(Tensor(name="grad_w5", shape=(1,), dtype="int64"))  # 8
    graph.add_tensor(Tensor(name="grad_w6", shape=None, dtype="bfloat16"))  # 2
    graph.add_tensor(Tensor(name="grad_w7", shape=("N", 10), dtype="float32"))  # 40
    graph.add_tensor(Tensor(name="grad_w8", shape=(1,), dtype="unknown_dtype"))  # 4 (default)

    graph.outputs.extend(
        [
            "grad_w1",
            "grad_w2",
            "grad_w3",
            "grad_w4",
            "grad_w5",
            "grad_w6",
            "grad_w7",
            "grad_w8",
            "all_gradients_flat",
        ]
    )

    # 400 + 10 + 4 + 8 + 8 + 2 + 40 + 4 = 476 bytes
    size = calculate_gradient_payload_size(graph)
    assert size == 476


def test_compress_gradients_int8():
    graph = Graph(name="g")
    compress_gradients_int8(graph)
    assert len(graph.nodes) == 0

    graph.outputs.append("grad_w1")
    compress_gradients_int8(graph)
    assert "grad_w1_quantized" in graph.outputs
    assert len(graph.nodes) == 1


def test_compile_multi_replica_graph():
    graph = Graph(name="g")
    assert compile_multi_replica_graph(graph, 1) is graph

    graph.initializers.append("w1")
    graph.add_tensor(Tensor(name="w1", shape=(10,), dtype="float32"))
    graph.inputs.append("w1")
    graph.inputs.append("x")
    graph.add_tensor(Tensor(name="x", shape=(10,), dtype="float32"))

    graph.add_node(Node("Add", ["w1", "x"], ["y"], name="add1"))
    graph.outputs.append("y")
    graph.add_tensor(Tensor(name="y", shape=(10,), dtype="float32", requires_grad=True))

    multi = compile_multi_replica_graph(graph, 2)
    assert "x_replica_0" in multi.inputs
    assert "x_replica_1" in multi.inputs
    assert "y_replica_0" in multi.outputs
    assert "y_replica_1" in multi.outputs
    assert "w1" in multi.initializers

    # testing out orig_out edge case branch
    graph2 = Graph(name="g2")
    graph2.inputs.append("x")
    graph2.add_node(Node("Identity", ["x"], ["y"], name="id1"))
    graph2.add_tensor(Tensor(name="y", shape=(1,), dtype="float32"))
    multi2 = compile_multi_replica_graph(graph2, 2)
    assert "y_replica_0" in multi2.tensors


def test_embed_distributed_identifiers():
    graph = Graph(name="g")
    graph.outputs.append("grad_w1")
    graph.add_tensor(Tensor(name="grad_w1", shape=(1,), dtype="float32"))
    embed_distributed_identifiers(graph)
    assert "distributed_sync_id_0" == graph.tensors["grad_w1"].doc_string


def test_add_synchronous_barrier():
    graph = Graph(name="g")
    graph.outputs.append("all_gradients_flat")
    add_synchronous_barrier(graph)
    assert "barrier_all_gradients_flat" in graph.outputs
    assert graph.nodes[0].op_type == "Identity"


def test_calculate_communication_bounds():
    graph = Graph(name="g")
    graph.outputs.append("grad_w1")
    graph.add_tensor(Tensor(name="grad_w1", shape=(1024, 1024), dtype="float32"))  # 4MB
    assert calculate_communication_bounds(graph, 100) == 40.0
    assert calculate_communication_bounds(graph, 0) == 0.0


def test_flatten_gradients():
    graph = Graph(name="g")
    flatten_gradients(graph)  # Should do nothing

    graph.outputs.extend(["grad_w1", "grad_w2"])
    flatten_gradients(graph)
    assert "all_gradients_flat" in graph.outputs


def test_reverse_topological_sort_cycle():
    graph = Graph(name="g")
    graph.add_node(Node("Id", ["x"], ["y"], name="n1"))
    graph.add_node(Node("Id", ["y"], ["x"], name="n2"))

    ordered = reverse_topological_sort(graph)
    assert len(ordered) == 2


def test_reverse_topological_sort_diamond():
    graph = Graph(name="g")
    graph.add_node(Node("Id", ["x"], ["y1"], name="n1"))
    graph.add_node(Node("Id", ["x"], ["y2"], name="n2"))
    graph.add_node(Node("Add", ["y1", "y2"], ["z"], name="n3"))
    ordered = reverse_topological_sort(graph)
    assert len(ordered) == 3
