"""Multi-device graph partitioning pass for ONNX9000 IR graphs."""

from onnx9000.core.ir import Graph


def partition_for_multi_device(graph: Graph, num_devices: int = 1) -> dict[str, Graph]:
    """Partition an ONNX9000 Graph across `num_devices` logical execution devices.

    Args:
        graph: The ONNX9000 Graph to partition.
        num_devices: Number of target devices (defaults to `1`).

    Returns:
        Dictionary mapping device identifier (`"device_0"`, `"device_1"`, ...) to its subgraph.
    """
    if num_devices <= 1 or not graph.nodes:
        return {"device_0": graph}

    partitions: dict[str, Graph] = {}
    chunk_size = max(1, (len(graph.nodes) + num_devices - 1) // num_devices)
    for dev_idx in range(num_devices):
        start = dev_idx * chunk_size
        end = min(len(graph.nodes), start + chunk_size)
        sub = Graph(f"{graph.name}_device_{dev_idx}")
        for node in graph.nodes[start:end]:
            sub.add_node(node)
        partitions[f"device_{dev_idx}"] = sub
    return partitions
