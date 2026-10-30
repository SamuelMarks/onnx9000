"""Memory consumption estimation and tensor lifecycle planning passes for ONNX9000."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import DynamicDim, Graph, Tensor


def _tensor_byte_size(tensor: Tensor) -> int:
    """Calculate the byte footprint of a single Tensor.

    Args:
        tensor: The Tensor to measure.

    Returns:
        Size in bytes.
    """
    dtype_sizes: dict[DType, int] = {
        DType.FLOAT32: 4,
        DType.FLOAT16: 2,
        DType.BFLOAT16: 2,
        DType.FLOAT64: 8,
        DType.INT32: 4,
        DType.INT64: 8,
        DType.INT16: 2,
        DType.INT8: 1,
        DType.UINT8: 1,
        DType.BOOL: 1,
    }
    itemsize = dtype_sizes.get(tensor.dtype, 4) if isinstance(tensor.dtype, DType) else 4
    count = 1
    if tensor.shape:
        for dim in tensor.shape:
            if isinstance(dim, int) and dim > 0:
                count *= dim
            elif isinstance(dim, DynamicDim) and isinstance(dim.value, int) and dim.value > 0:
                count *= dim.value
    return count * itemsize


def estimate_memory_consumption(graph: Graph) -> dict[str, int]:
    """Estimate per-tensor memory footprint in bytes for all registered tensors in `graph`.

    Args:
        graph: The ONNX9000 Graph to analyze.

    Returns:
        Dictionary mapping tensor name to estimated byte size (empty if graph has no tensors).
    """
    if not graph.tensors:
        return {}
    return {name: _tensor_byte_size(tensor) for name, tensor in graph.tensors.items()}


def plan_tensor_lifecycles(graph: Graph) -> dict[str, tuple[int, int]]:
    """Compute `(first_step, last_step)` node-index lifespans for intermediate tensors.

    Args:
        graph: The ONNX9000 Graph to analyze.

    Returns:
        Dictionary mapping tensor name to `(birth_node_idx, death_node_idx)` tuple.
    """
    if not graph.nodes:
        return {}
    lifespans: dict[str, tuple[int, int]] = {}
    for idx, node in enumerate(graph.nodes):
        for out in node.outputs:
            out_name = out.name if isinstance(out, Tensor) else str(out)
            if out_name not in lifespans:
                lifespans[out_name] = (idx, idx)
        for inp in node.inputs:
            inp_name = inp.name if isinstance(inp, Tensor) else str(inp)
            if inp_name in lifespans:
                start, _ = lifespans[inp_name]
                lifespans[inp_name] = (start, idx)
            else:
                lifespans[inp_name] = (0, idx)
    return lifespans
