"""Module providing core logic and structural definitions."""

import hashlib

from onnx9000.core.ir import Graph


def hash_graph(graph: Graph) -> str:
    """
    Computes a SHA-256 hash over the structural content of the Graph and its weights.
    Used for cache key generation.
    """
    hasher = hashlib.sha256()

    # Hash graph structural info
    hasher.update(graph.name.encode("utf-8"))

    # Hash nodes
    for node in graph.nodes:
        hasher.update(node.op_type.encode("utf-8"))
        for inp in node.inputs:
            hasher.update(inp.encode("utf-8"))
        for out in node.outputs:
            hasher.update(out.encode("utf-8"))
        # Could serialize attributes here

    # Hash shapes and dtypes
    for name, tensor in graph.tensors.items():
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tensor.shape).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))

        # If it's an initializer, optionally hash the weights.
        # Note: In a real system, you might not hash weights for JIT if weights can be loaded dynamically,
        # but if we embed or rely on specific weight shapes, we do.
        if tensor.is_initializer and tensor.data is not None:
            # Only hash a sample of weights for speed, or a checksum
            hasher.update(tensor.data.tobytes()[:1024])

    return hasher.hexdigest()
