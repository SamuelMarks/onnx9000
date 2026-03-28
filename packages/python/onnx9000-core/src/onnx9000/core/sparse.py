"""Module for sparse tensor format conversions and parsers."""

import json
import logging
import struct
from typing import Any, Optional, Union

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, SparseTensor

logger = logging.getLogger(__name__)


def get_struct_fmt(dtype: DType) -> str:
    """Get the struct format character for a given DType."""
    mapping = {
        DType.FLOAT32: "f",
        DType.FLOAT64: "d",
        DType.INT8: "b",
        DType.INT16: "h",
        DType.INT32: "i",
        DType.INT64: "q",
        DType.UINT8: "B",
        DType.UINT16: "H",
        DType.UINT32: "I",
        DType.UINT64: "Q",
        DType.BOOL: "?",
    }
    # FLOAT16 ('e') is supported in Python 3.6+
    try:
        if dtype == DType.FLOAT16:
            return "e"
    except (AttributeError, ValueError):
        pass
    return mapping.get(dtype, "B")


def get_byte_size(dtype: DType) -> int:
    """Get the byte size for a given DType."""
    fmt = get_struct_fmt(dtype)
    return struct.calcsize(fmt)


def unpack_data(data: bytes, dtype: DType) -> list[Any]:
    """Unpack raw bytes into a list of Python values based on DType."""
    if not data:
        return []
    fmt = get_struct_fmt(dtype)
    size = struct.calcsize(fmt)
    count = len(data) // size
    return list(struct.unpack(f"<{count}{fmt}", data))


def pack_data(values: list[Any], dtype: DType) -> bytes:
    """Pack a list of Python values into raw bytes based on DType."""
    if not values:
        return b""

    target_dtype = dtype
    if dtype == DType.FLOAT64:
        logger.info("Downcasting float64 values to float32 for compatibility.")
        values = [float(v) for v in values]
        target_dtype = DType.FLOAT32

    fmt = get_struct_fmt(target_dtype)
    count = len(values)
    return struct.pack(f"<{count}{fmt}", *values)


def pack_sparse_int8(values: list[int], indices: list[int]) -> bytes:
    """Item 80: Compress Sparse INT8 tensors via bit-packing (storing 4-bit indices and 8-bit values).

    Format: Each entry is 12 bits. Packed into bytes.
    [index_4bit][value_8bit]
    """
    packed = bytearray()
    for i in range(0, len(values), 2):
        # Pack two entries (24 bits = 3 bytes)
        # Entry 1: [idx1:4][val1:8]
        # Entry 2: [idx2:4][val2:8]
        if i + 1 < len(values):
            idx1 = indices[i] & 0x0F
            val1 = values[i] & 0xFF
            idx2 = indices[i + 1] & 0x0F
            val2 = values[i + 1] & 0xFF

            # byte 1: [idx1:4][val1_high:4]
            # byte 2: [val1_low:4][idx2:4]
            # byte 3: [val2:8]
            b1 = (idx1 << 4) | (val1 >> 4)
            b2 = ((val1 & 0x0F) << 4) | idx2
            b3 = val2
            packed.extend([b1, b2, b3])
        else:
            # Last entry if odd
            idx1 = indices[i] & 0x0F
            val1 = values[i] & 0xFF
            b1 = (idx1 << 4) | (val1 >> 4)
            b2 = (val1 & 0x0F) << 4
            packed.extend([b1, b2])

    return bytes(packed)


def dense_to_coo(tensor: Constant) -> SparseTensor:
    """Convert a dense Constant to a COO (Coordinate List) SparseTensor."""
    if tensor.data is None:
        return SparseTensor(tensor.name, dims=tensor.shape, format="COO")

    values = unpack_data(tensor.data, tensor.dtype)
    non_zero_values = []
    non_zero_indices = []

    for i, val in enumerate(values):
        if val != 0:
            non_zero_values.append(val)
            non_zero_indices.append(i)

    val_const = Constant(
        f"{tensor.name}_values",
        values=pack_data(non_zero_values, tensor.dtype),
        shape=(len(non_zero_values),),
        dtype=tensor.dtype,
    )
    idx_const = Constant(
        f"{tensor.name}_indices",
        values=pack_data(non_zero_indices, DType.INT64),
        shape=(len(non_zero_indices),),
        dtype=DType.INT64,
    )

    return SparseTensor(
        tensor.name,
        values=val_const,
        indices=idx_const,
        dims=tensor.shape,
        format="COO",
    )


def dense_to_csr(tensor: Constant) -> SparseTensor:
    """Convert a dense Constant to a CSR (Compressed Sparse Row) SparseTensor."""
    if len(tensor.shape) != 2:
        return dense_to_coo(tensor)

    rows, cols = tensor.shape

    if rows * cols < 1024:
        logger.warning(
            f"Tensor '{tensor.name}' is small ({rows}x{cols}). CSR overhead might outweigh dense execution."
        )

    if rows > 2**31 - 1 or cols > 2**31 - 1:
        raise ValueError(f"Tensor dimensions {tensor.shape} exceed INT32 limits for CSR indexing.")

    values = unpack_data(tensor.data, tensor.dtype)

    csr_values = []
    csr_col_indices = []
    csr_row_ptr = [0]

    for r in range(rows):
        count = 0
        for c in range(cols):
            idx = r * cols + c
            val = values[idx]
            if val != 0:
                csr_values.append(val)
                csr_col_indices.append(c)
                count += 1
        csr_row_ptr.append(csr_row_ptr[-1] + count)

    val_const = Constant(
        f"{tensor.name}_values",
        values=pack_data(csr_values, tensor.dtype),
        shape=(len(csr_values),),
        dtype=tensor.dtype,
    )
    col_idx_const = Constant(
        f"{tensor.name}_col_indices",
        values=pack_data(csr_col_indices, DType.INT64),
        shape=(len(csr_col_indices),),
        dtype=DType.INT64,
    )
    row_ptr_const = Constant(
        f"{tensor.name}_row_ptr",
        values=pack_data(csr_row_ptr, DType.INT64),
        shape=(len(csr_row_ptr),),
        dtype=DType.INT64,
    )

    return SparseTensor(
        tensor.name,
        values=val_const,
        col_indices=col_idx_const,
        row_ptr=row_ptr_const,
        dims=tensor.shape,
        format="CSR",
    )


def dense_to_csc(tensor: Constant) -> SparseTensor:
    """Convert a dense Constant to a CSC (Compressed Sparse Column) SparseTensor."""
    if len(tensor.shape) != 2:
        return dense_to_coo(tensor)

    rows, cols = tensor.shape
    values = unpack_data(tensor.data, tensor.dtype)

    csc_values = []
    csc_row_indices = []
    csc_col_ptr = [0]

    for c in range(cols):
        count = 0
        for r in range(rows):
            idx = r * cols + c
            val = values[idx]
            if val != 0:
                csc_values.append(val)
                csc_row_indices.append(r)
                count += 1
        csc_col_ptr.append(csc_col_ptr[-1] + count)

    val_const = Constant(
        f"{tensor.name}_values",
        values=pack_data(csc_values, tensor.dtype),
        shape=(len(csc_values),),
        dtype=tensor.dtype,
    )
    row_idx_const = Constant(
        f"{tensor.name}_row_indices",
        values=pack_data(csc_row_indices, DType.INT64),
        shape=(len(csc_row_indices),),
        dtype=DType.INT64,
    )
    col_ptr_const = Constant(
        f"{tensor.name}_col_ptr",
        values=pack_data(csc_col_ptr, DType.INT64),
        shape=(len(csc_col_ptr),),
        dtype=DType.INT64,
    )

    return SparseTensor(
        tensor.name,
        values=val_const,
        col_indices=row_idx_const,
        row_ptr=col_ptr_const,
        dims=tensor.shape,
        format="CSC",
    )


def dense_to_bsr(tensor: Constant, block_size: tuple[int, int]) -> SparseTensor:
    """Convert a dense Constant to a BSR (Block Sparse Row) SparseTensor."""
    if len(tensor.shape) != 2:
        return dense_to_coo(tensor)

    rows, cols = tensor.shape
    bh, bw = block_size

    if rows % bh != 0 or cols % bw != 0:
        return dense_to_coo(tensor)

    values = unpack_data(tensor.data, tensor.dtype)

    bsr_values = []
    bsr_col_indices = []
    bsr_row_ptr = [0]

    for rb in range(rows // bh):
        count = 0
        for cb in range(cols // bw):
            block = []
            is_non_zero = False
            for r in range(rb * bh, (rb + 1) * bh):
                for c in range(cb * bw, (cb + 1) * bw):
                    val = values[r * cols + c]
                    block.append(val)
                    if val != 0:
                        is_non_zero = True
            if is_non_zero:
                bsr_values.extend(block)
                bsr_col_indices.append(cb)
                count += 1
        bsr_row_ptr.append(bsr_row_ptr[-1] + count)

    val_const = Constant(
        f"{tensor.name}_values",
        values=pack_data(bsr_values, tensor.dtype),
        shape=(len(bsr_values),),
        dtype=tensor.dtype,
    )
    col_idx_const = Constant(
        f"{tensor.name}_col_indices",
        values=pack_data(bsr_col_indices, DType.INT64),
        shape=(len(bsr_col_indices),),
        dtype=DType.INT64,
    )
    row_ptr_const = Constant(
        f"{tensor.name}_row_ptr",
        values=pack_data(bsr_row_ptr, DType.INT64),
        shape=(len(bsr_row_ptr),),
        dtype=DType.INT64,
    )

    return SparseTensor(
        tensor.name,
        values=val_const,
        col_indices=col_idx_const,
        row_ptr=row_ptr_const,
        dims=tensor.shape,
        block_dims=block_size,
        format="BSR",
    )


def sparse_to_coo(sparse_tensor: SparseTensor) -> SparseTensor:
    """Convert any SparseTensor to COO (Coordinate List) format."""
    if sparse_tensor.format == "COO":
        return sparse_tensor

    dims = sparse_tensor.shape
    if sparse_tensor.format == "CSR":
        csr_values = unpack_data(sparse_tensor.values.data, sparse_tensor.values.dtype)
        csr_col_indices = unpack_data(sparse_tensor.col_indices.data, DType.INT64)
        csr_row_ptr = unpack_data(sparse_tensor.row_ptr.data, DType.INT64)

        coo_values = csr_values
        coo_indices = []
        rows, cols = dims
        for r in range(rows):
            for i in range(csr_row_ptr[r], csr_row_ptr[r + 1]):
                c = csr_col_indices[i]
                coo_indices.append(r * cols + c)

        val_const = Constant(
            f"{sparse_tensor.name}_values",
            values=pack_data(coo_values, sparse_tensor.values.dtype),
            shape=(len(coo_values),),
            dtype=sparse_tensor.values.dtype,
        )
        idx_const = Constant(
            f"{sparse_tensor.name}_indices",
            values=pack_data(coo_indices, DType.INT64),
            shape=(len(coo_indices),),
            dtype=DType.INT64,
        )
        return SparseTensor(
            sparse_tensor.name, values=val_const, indices=idx_const, dims=dims, format="COO"
        )

    if sparse_tensor.format == "CSC":
        csc_values = unpack_data(sparse_tensor.values.data, sparse_tensor.values.dtype)
        csc_row_indices = unpack_data(sparse_tensor.col_indices.data, DType.INT64)
        csc_col_ptr = unpack_data(sparse_tensor.row_ptr.data, DType.INT64)

        coo_values = csc_values
        coo_indices = []
        rows, cols = dims
        for c in range(cols):
            for i in range(csc_col_ptr[c], csc_col_ptr[c + 1]):
                r = csc_row_indices[i]
                coo_indices.append(r * cols + c)

        val_const = Constant(
            f"{sparse_tensor.name}_values",
            values=pack_data(coo_values, sparse_tensor.values.dtype),
            shape=(len(coo_values),),
            dtype=sparse_tensor.values.dtype,
        )
        idx_const = Constant(
            f"{sparse_tensor.name}_indices",
            values=pack_data(coo_indices, DType.INT64),
            shape=(len(coo_indices),),
            dtype=DType.INT64,
        )
        return SparseTensor(
            sparse_tensor.name, values=val_const, indices=idx_const, dims=dims, format="COO"
        )

    if sparse_tensor.format == "BSR":
        bsr_values = unpack_data(sparse_tensor.values.data, sparse_tensor.values.dtype)
        bsr_col_indices = unpack_data(sparse_tensor.col_indices.data, DType.INT64)
        bsr_row_ptr = unpack_data(sparse_tensor.row_ptr.data, DType.INT64)
        bh, bw = sparse_tensor.block_dims

        coo_values = []
        coo_indices = []
        rows, cols = dims

        block_idx = 0
        for rb in range(rows // bh):
            for i in range(bsr_row_ptr[rb], bsr_row_ptr[rb + 1]):
                cb = bsr_col_indices[i]
                for r in range(bh):
                    for c in range(bw):
                        val = bsr_values[block_idx * bh * bw + r * bw + c]
                        if val != 0:
                            coo_indices.append((rb * bh + r) * cols + (cb * bw + c))
                            coo_values.append(val)
                block_idx += 1

        val_const = Constant(
            f"{sparse_tensor.name}_values",
            values=pack_data(coo_values, sparse_tensor.values.dtype),
            shape=(len(coo_values),),
            dtype=sparse_tensor.values.dtype,
        )
        idx_const = Constant(
            f"{sparse_tensor.name}_indices",
            values=pack_data(coo_indices, DType.INT64),
            shape=(len(coo_indices),),
            dtype=DType.INT64,
        )
        return SparseTensor(
            sparse_tensor.name, values=val_const, indices=idx_const, dims=dims, format="COO"
        )

    return sparse_tensor


def sparse_to_dense(sparse_tensor: SparseTensor) -> Constant:
    """Convert any SparseTensor to a dense Constant."""
    coo = sparse_to_coo(sparse_tensor)
    if coo.values is None or coo.indices is None:
        return Constant(coo.name, shape=coo.shape, dtype=DType.FLOAT32)

    values = unpack_data(coo.values.data, coo.values.dtype)
    indices = unpack_data(coo.indices.data, DType.INT64)

    total_size = 1
    for dim in coo.shape:
        total_size *= int(dim.value) if hasattr(dim, "value") else int(dim)

    dense_values = [0] * total_size
    for idx, val in zip(indices, values):
        dense_values[idx] = val

    return Constant(
        coo.name,
        values=pack_data(dense_values, coo.values.dtype),
        shape=coo.shape,
        dtype=coo.values.dtype,
    )


def detect_theoretical_sparsity(tensor: Constant, epsilon: float = 1e-06) -> float:
    """Detect maximum sparsity theoretically achievable based on epsilon values."""
    if tensor.data is None:
        return 1.0
    values = unpack_data(tensor.data, tensor.dtype)
    if not values:
        return 1.0
    zeros = sum(1 for v in values if abs(v) < epsilon)
    return zeros / len(values)


def calculate_memory_usage(tensor: Union[Constant, SparseTensor]) -> int:
    """Provide memory usage calculation (Dense vs Sparse byte comparison)."""
    if isinstance(tensor, Constant):
        return len(tensor.data) if tensor.data else 0
    usage = 0
    if tensor.values:
        usage += len(tensor.values.data) if tensor.values.data else 0
    if tensor.indices:
        usage += len(tensor.indices.data) if tensor.indices.data else 0
    if tensor.row_ptr:
        usage += len(tensor.row_ptr.data) if tensor.row_ptr.data else 0
    if tensor.col_indices:
        usage += len(tensor.col_indices.data) if tensor.col_indices.data else 0
    return usage


def compression_ratio(dense: Constant, sparse: SparseTensor) -> float:
    """Track compression ratio mathematically (1.0 - (sparse_size / dense_size))."""
    dense_size = calculate_memory_usage(dense)
    sparse_size = calculate_memory_usage(sparse)
    if dense_size == 0:
        return 0.0
    return 1.0 - (sparse_size / dense_size)


def profile(graph: Graph) -> dict:
    """Profile model sparsity and theoretical savings."""
    report = {
        "layers": [],
        "total_dense_size": 0,
        "total_sparse_size": 0,
        "overall_sparsity": 0.0,
    }

    if not graph.tensors:
        return report

    for name, tensor in graph.tensors.items():
        if isinstance(tensor, (Constant, SparseTensor)):
            total_elements = 1
            if tensor.shape:
                for dim in tensor.shape:
                    total_elements *= int(dim.value) if hasattr(dim, "value") else int(dim)

            theoretical_dense_size = total_elements * get_byte_size(tensor.dtype)
            current_size = calculate_memory_usage(tensor)

            sparsity = (
                1.0 - (current_size / theoretical_dense_size) if theoretical_dense_size > 0 else 0.0
            )

            report["layers"].append(
                {
                    "name": name,
                    "dense_bytes": theoretical_dense_size,
                    "sparse_bytes": current_size,
                    "sparsity": sparsity,
                }
            )

            report["total_dense_size"] += theoretical_dense_size
            report["total_sparse_size"] += current_size

    if report["total_dense_size"] > 0:
        report["overall_sparsity"] = 1.0 - (
            report["total_sparse_size"] / report["total_dense_size"]
        )

    return report


def get_sparsity_report(graph: Graph) -> str:
    """Render layer-by-layer sparsity percentage in an ASCII table."""
    data = profile(graph)
    if not data["layers"]:
        return "No trainable parameters found."

    lines = []
    lines.append(f"{'Layer Name':<40} | {'Sparsity':<10} | {'Saving (KB)':<12}")
    lines.append("-" * 70)
    for layer in data["layers"]:
        saving_kb = (layer["dense_bytes"] - layer["sparse_bytes"]) / 1024
        lines.append(f"{layer['name']:<40} | {layer['sparsity']:>9.2%} | {saving_kb:>12.2f}")
    lines.append("-" * 70)
    total_saving_mb = (data["total_dense_size"] - data["total_sparse_size"]) / (1024 * 1024)
    lines.append(f"{'OVERALL':<40} | {data['overall_sparsity']:>9.2%} | {total_saving_mb:>9.2f} MB")
    return "\n".join(lines)


def generate_json_report(graph: Graph, output_path: str) -> None:
    """Item 223: Output a detailed JSON 'Sparsity Report Card' suitable for MLOps dashboards."""
    data = profile(graph)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Sparsity report card saved to {output_path}")


def evaluate_mse(original: Constant, pruned: Union[Constant, SparseTensor]) -> float:
    """Evaluate Mean Squared Error (MSE) degradation after applying a sparse mask."""
    orig_vals = unpack_data(original.data, original.dtype)

    if isinstance(pruned, SparseTensor):
        pruned_dense = sparse_to_dense(pruned)
        pruned_vals = unpack_data(pruned_dense.data, pruned_dense.dtype)
    else:
        pruned_vals = unpack_data(pruned.data, pruned.dtype)

    if not orig_vals:
        return 0.0

    mse = sum((o - p) ** 2 for o, p in zip(orig_vals, pruned_vals)) / len(orig_vals)
    return float(mse)


def de_sparsify(graph: Graph) -> None:
    """Inflate SparseTensor back into dense Constant arrays for backwards compatibility."""
    for name in list(graph.tensors.keys()):
        tensor = graph.tensors[name]
        if isinstance(tensor, SparseTensor):
            dense_tensor = sparse_to_dense(tensor)
            graph.tensors[name] = dense_tensor
            if name in graph.sparse_initializers:
                graph.sparse_initializers.remove(name)
            if name not in graph.initializers:
                graph.initializers.append(name)


def convert_hf_sparse_to_onnx(graph: Graph) -> None:
    """Item 15: Support converting standard HuggingFace sparse models to ONNX native sparse models."""
    for name, tensor in list(graph.tensors.items()):
        is_hf_sparse = False
        if hasattr(tensor, "metadata_props"):
            if "sparse_method" in tensor.metadata_props or "bitmask" in name:
                is_hf_sparse = True

        if is_hf_sparse and isinstance(tensor, Constant):
            sparse_tensor = dense_to_coo(tensor)
            graph.tensors[name] = sparse_tensor
            if name in graph.initializers:
                graph.initializers.remove(name)
            if name not in graph.sparse_initializers:
                graph.sparse_initializers.append(name)


def resolve_nm_metadata(graph: Graph) -> None:
    """Item 64: Resolve 2:4 sparse encoding metadata specifically for TensorRT / WebGPU injection."""
    for name, tensor in graph.tensors.items():
        if isinstance(tensor, SparseTensor) and tensor.format == "NM":
            if not hasattr(tensor, "metadata_props"):
                tensor.metadata_props = {}
            tensor.metadata_props["sparse_encoding"] = "2:4"
            tensor.metadata_props["backend_injection"] = "tensorrt_webgpu"


def map_sparse_to_safetensors(graph: Graph) -> None:
    """Item 18: Support mapping sparse inputs correctly to safetensors external references."""
    for name, tensor in graph.tensors.items():
        if isinstance(tensor, SparseTensor):
            if not hasattr(tensor, "metadata_props"):
                tensor.metadata_props = {}
            tensor.metadata_props["safetensors_reference"] = f"sparse/{name}"
            if tensor.values:
                if not hasattr(tensor.values, "metadata_props"):
                    tensor.values.metadata_props = {}
                tensor.values.metadata_props["safetensors_reference"] = f"sparse/{name}/values"
            if tensor.indices:
                if not hasattr(tensor.indices, "metadata_props"):
                    tensor.indices.metadata_props = {}
                tensor.indices.metadata_props["safetensors_reference"] = f"sparse/{name}/indices"


def validate_provider_support(graph: Graph, providers: list[str]) -> None:
    """Item 227: Emit warnings if a generated sparse model is loaded into an older execution provider lacking sparse support."""
    has_sparse = any(isinstance(t, SparseTensor) for t in graph.tensors.values())
    if has_sparse:
        sparse_ready_providers = ["WebGPU", "WASM_SIMD", "CPU"]
        for p in providers:
            if p not in sparse_ready_providers:
                logger.warning(
                    f"Provider '{p}' might not natively support SparseTensor format. Performance may be degraded or execution might fail."
                )


def collapse_sparse_tensors(graph: Graph) -> None:
    """Identify and collapse structurally 100% sparse Constant tensors into a scalar 0."""
    for name in list(graph.tensors.keys()):
        tensor = graph.tensors[name]
        if isinstance(tensor, (Constant, SparseTensor)):
            is_all_zero = False
            if isinstance(tensor, Constant) and tensor.data:
                values = unpack_data(tensor.data, tensor.dtype)
                if all(v == 0 for v in values):
                    is_all_zero = True
            elif isinstance(tensor, SparseTensor):
                if not tensor.values or not tensor.values.data:
                    is_all_zero = True
                else:
                    values = unpack_data(tensor.values.data, tensor.values.dtype)
                    if all(v == 0 for v in values):
                        is_all_zero = True

            if is_all_zero:
                if isinstance(tensor, Constant):
                    tensor.data = pack_data([0], tensor.dtype)
                    tensor.shape = (1,)
                elif isinstance(tensor, SparseTensor):
                    graph.tensors[name] = Constant(
                        name, values=pack_data([0], tensor.dtype), shape=(1,), dtype=tensor.dtype
                    )
                    if name in graph.sparse_initializers:
                        graph.sparse_initializers.remove(name)
                    if name not in graph.initializers:
                        graph.initializers.append(name)


def strip_dense_representation(graph: Graph, tensor_name: str) -> None:
    """Strip the dense representation immediately to trigger JS Garbage Collection."""
    if tensor_name in graph.tensors:
        tensor = graph.tensors[tensor_name]
        if isinstance(tensor, Constant):
            tensor.data = None


def analyze_topological_dead_ends(graph: Graph) -> list[str]:
    """Analyze topological dead ends created by 100% sparse layers."""
    dead_nodes = []
    dead_tensors = set()

    for name, tensor in graph.tensors.items():
        if isinstance(tensor, (Constant, SparseTensor)):
            is_all_zero = False
            if isinstance(tensor, Constant) and tensor.data:
                values = unpack_data(tensor.data, tensor.dtype)
                if all(v == 0 for v in values):
                    is_all_zero = True
            elif isinstance(tensor, SparseTensor):
                if not tensor.values or not tensor.values.data:
                    is_all_zero = True
                else:
                    values = unpack_data(tensor.values.data, tensor.values.dtype)
                    if all(v == 0 for v in values):
                        is_all_zero = True
            if is_all_zero:
                dead_tensors.add(name)

    from onnx9000.core.utils import topological_sort

    sorted_nodes = topological_sort(graph)
    for node in sorted_nodes:
        is_dead = False
        if node.op_type in ["MatMul", "Conv", "Gemm"]:
            if len(node.inputs) > 1 and node.inputs[1] in dead_tensors:
                is_dead = True
        elif node.op_type in ["Relu", "Sigmoid", "Tanh"]:
            if node.inputs[0] in dead_tensors:
                is_dead = True

        if is_dead:
            dead_nodes.append(node.name)
            for out in node.outputs:
                dead_tensors.add(out)

    return dead_nodes
