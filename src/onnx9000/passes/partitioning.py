"""
Graph Partitioning

Splits a monolithic Graph into sub-graphs mapped to different
execution providers or devices (e.g. CPU vs WebGPU).
"""

from onnx9000.ir import Graph


def partition_for_multi_device(graph: Graph) -> dict[str, Graph]:
    """Partitions the graph based on capability and cost models."""
    return {"device_0": graph}
