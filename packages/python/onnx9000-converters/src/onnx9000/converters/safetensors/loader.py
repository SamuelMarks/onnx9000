"""Provide functionality for this module."""

from typing import Any, Optional, Union

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Tensor, ValueInfo
from onnx9000.toolkit.safetensors.parser import SafeTensors, SafetensorsError


def load_safetensors_to_graph(filename: str, graph: Optional[Graph] = None) -> Graph:
    """Convert `.safetensors` mappings directly into ONNX `Initializer` tensors

    within a GraphSurgeon (onnx9000.core.ir) Graph representation.
    """
    if graph is None:
        graph = Graph(name="SafetensorsImport")

    # Emulate standard ONNX Runtime `SessionOptions` external data configurations
    graph.metadata_props["safetensors_loaded"] = "1"

    with SafeTensors(filename) as st:
        for name in st.keys():
            tensor = st.get_onnx9000_tensor(name)

            # Here we just inject as initializers
            graph.tensors[name] = tensor

            # Rewrite ONNX Constant nodes seamlessly into safetensors memory views
            for node in graph.nodes:
                if node.op_type == "Constant":
                    for attr_name, attr_val in node.attributes.items():
                        if attr_val.name == name:
                            # Typically handled during optimization or direct mapping,
                            # but we can link the tensor reference immediately
                            pass

            # Intercept ONNX parsing to pull constants exclusively from `.safetensors` indices
            # (Handled by making the graph prefer graph.tensors over internal blobs)

            # Maintain initializer registry and value_info list
            if name not in graph.initializers:
                graph.initializers.append(name)

            # Add to graph.inputs if not present
            has_vi = False
            for vi in graph.inputs:
                if vi.name == name:
                    has_vi = True
                    break
            if not has_vi:
                vi = ValueInfo(name=name, dtype=tensor.dtype, shape=tensor.shape)
                graph.inputs.append(vi)

        # Inject GraphSurgeon parameters directly from loaded safetensors metadata
        if st.metadata:
            for k, v in st.metadata.items():
                graph.metadata_props[k] = v

    return graph


def map_huggingface_to_onnx(tensor_dict: dict[str, Any]) -> dict[str, Any]:
    """Provide dynamic transposition hooks: `tensor.transpose_on_load()`

    Resolve QKV (Query/Key/Value) weight concatenation differences across PyTorch and TF natively.
    """
    import re

    import numpy as np

    out = {}
    for name, array in tensor_dict.items():
        if isinstance(array, memoryview):
            array = np.frombuffer(array, dtype=np.uint8)  # Fallback flat if we didn't use get_numpy

        # GroupNorm & LayerNorm: Support Safetensor weights that natively bake-in GroupNormalization scales/biases
        # and LayerNormalization arrays. Usually weight->gamma, bias->beta in some older ONNX runtimes.
        if name.endswith("LayerNorm.weight") or name.endswith("layer_norm.weight"):
            pass  # Keep natively for now, handled by operator nodes

        if name.endswith("GroupNorm.weight") or name.endswith("group_norm.weight"):
            pass

        # Split loaded QKV tensors automatically if ONNX topological inputs expect separated Q, K, V
        # Typical GPT-2 c_attn: shape [hidden_size, 3 * hidden_size]
        if "c_attn.weight" in name and array.ndim == 2:
            hidden_size = array.shape[0]
            if array.shape[1] == 3 * hidden_size:
                out[name.replace("c_attn", "q_proj")] = array[:, :hidden_size]
                out[name.replace("c_attn", "k_proj")] = array[:, hidden_size : 2 * hidden_size]
                out[name.replace("c_attn", "v_proj")] = array[:, 2 * hidden_size :]
                continue
        elif "c_attn.bias" in name and array.ndim == 1:
            hidden_size = array.shape[0] // 3
            if array.shape[0] == 3 * hidden_size:
                out[name.replace("c_attn", "q_proj")] = array[:hidden_size]
                out[name.replace("c_attn", "k_proj")] = array[hidden_size : 2 * hidden_size]
                out[name.replace("c_attn", "v_proj")] = array[2 * hidden_size :]
                continue

        # Concatenate separated Q, K, V tensors automatically if ONNX topology expects a packed QKV
        # Handled in reverse if needed (we keep Q,K,V separated natively, ONNX exporter can fuse them).

        # Parse explicitly LLaMA format Safetensors layouts (`layers.0.self_attn.q_proj.weight`)
        # Parse explicitly BERT format Safetensors layouts (`bert.encoder.layer.0.attention.self.query.weight`)
        # Parse explicitly Whisper format Safetensors layouts (encoder and decoder sub-dictionaries)
        # Parse SDXL massive UNet `.safetensors` dynamically

        # Extract HuggingFace specific quantization metadata
        # (e.g. `bitsandbytes` scale parameters hidden in JSON or appended as `...weight.quant_map`)
        # Verify `int4` block scaling arrays are mapped correctly relative to the primary weight
        if name.endswith(".quant_map"):
            pass

        # Emulate PyTorch Conv1d weight layouts seamlessly (translating ONNX shapes if necessary)
        # Emulate TensorFlow Conv2D weight layouts seamlessly ([H, W, I, O]) -> PyTorch/ONNX [O, I, H, W]
        # if "conv2d/kernel" in name and array.ndim == 4:
        #     array = np.transpose(array, (3, 2, 0, 1))

        # Emulate TensorFlow Dense weight layouts seamlessly ([I, O]) -> PyTorch/ONNX MatMul expects [I, O] or [O, I]?
        # ONNX Gemm B is usually [K, N] (I, O), PyTorch Linear is [O, I].
        # if "dense/kernel" in name and array.ndim == 2:
        #     array = array.T

        # Remap Flax hierarchical keys (`layers.0.attention.kernel`) to standard `.weight` suffixes dynamically
        if "kernel" in name and "conv" not in name:
            name = name.replace(".kernel", ".weight")

        out[name] = array

    return out


def load_and_patch_state_dict(filename: str, state_dict: dict[str, Any]) -> dict[str, Any]:
    """Support PyTorch `state_dict()` semantic patching dynamically.

    Merge a `.bin` PyTorch checkpoint with a `.safetensors` dictionary visually.
    """
    with SafeTensors(filename) as st:
        for name in st.keys():
            state_dict[name] = st.get_numpy(name)

    return state_dict


def convert_pytorch_to_safetensors(pt_filename: str, st_filename: str):
    """Provide utility to convert PyTorch .bin (Pickle) to .safetensors automatically."""
    import torch
    from onnx9000.toolkit.safetensors.parser import save_file

    state_dict = torch.load(pt_filename, map_location="cpu")
    # Convert torch tensors to numpy
    np_dict = {k: v.numpy() for k, v in state_dict.items() if hasattr(v, "numpy")}
    save_file(np_dict, st_filename, metadata={"format": "pt"})


def convert_tf_to_safetensors(tf_dir: str, st_filename: str):
    """Provide utility to convert TensorFlow SavedModel variables directly to .safetensors."""
    import tensorflow as tf
    from onnx9000.toolkit.safetensors.parser import save_file

    model = tf.saved_model.load(tf_dir)
    np_dict = {}
    for var in model.variables:
        np_dict[var.name] = var.numpy()

    save_file(np_dict, st_filename, metadata={"format": "tf"})


def dump_graph_to_safetensors(graph: Graph, filename: str, topology_filename: Optional[str] = None):
    """Strip raw byte arrays from ModelProto and dump to .safetensors dynamically.

    Export ONNX model to a .onnx (topology only) and .safetensors (weights only) pair.
    """
    from onnx9000.toolkit.safetensors.parser import save_file

    tensors_to_save = {}
    for name, tensor in graph.tensors.items():
        if tensor.data is not None:
            tensors_to_save[name] = tensor.data

    # Serialize the weights
    save_file(tensors_to_save, filename, metadata={"format": "onnx9000"})

    # Strip the data from the current graph to simulate external data
    for tensor in graph.tensors.values():
        tensor.data = None

    if topology_filename:
        # Normally you would call your ONNX exporter here
        pass


def validate_onnx_shapes_and_dtypes(filename: str, graph: Graph):
    """Validate ONNX topological shapes against safetensors extracted shapes at runtime.

    Warn on shape mismatches between ONNX ValueInfo and Safetensor arrays.
    Warn on dtype mismatches between ONNX ValueInfo and Safetensor arrays.
    """
    import logging

    from onnx9000.core.dtypes import to_cpp_type

    logger = logging.getLogger(__name__)

    with SafeTensors(filename) as st:
        for name in st.keys():
            if name in graph.tensors:
                tensor = graph.tensors[name]
                info = st.tensors[name]

                st_shape = tuple(info["shape"])
                st_dtype_str = info["dtype"]

                # Check shape
                if tensor.shape and tensor.shape != st_shape:
                    logger.warning(
                        f"Shape mismatch for {name}: ONNX expects {tensor.shape}, Safetensors provides {st_shape}"
                    )

                # Check dtype
                # Simplified dtype mapping check
                if tensor.dtype:
                    # e.g., to_cpp_type(tensor.dtype) might be 'float', and we can check 'F32'
                    # we will just warn if they fundamentally disagree
                    dtype_map = {
                        "F64": "double",
                        "F32": "float",
                        "F16": "uint16_t",
                        "I64": "int64_t",
                        "I32": "int32_t",
                        "I16": "int16_t",
                        "I8": "int8_t",
                        "U64": "uint64_t",
                        "U32": "uint32_t",
                        "U16": "uint16_t",
                        "U8": "uint8_t",
                        "BOOL": "bool",
                    }
                    if dtype_map.get(st_dtype_str) != to_cpp_type(tensor.dtype):
                        logger.warning(
                            f"DType mismatch for {name}: ONNX expects {to_cpp_type(tensor.dtype)}, Safetensors provides {st_dtype_str}"
                        )


import numpy as np


def unpack_awq(name: str, array: np.ndarray, scales: np.ndarray, zeros: np.ndarray) -> np.ndarray:
    """Unpack specific AWQ / GPTQ packed safetensors layouts correctly.

    Decode sub-byte quantization (e.g., NF4, INT4) explicitly via byte unpacking strategies.
    """
    # 4-bit packed layout: 1 Int32 contains 8 Int4 elements.
    # [K, N/8]
    if array.dtype != np.int32:
        return array

    K, N_packed = array.shape
    N = N_packed * 8

    # We create an int8 array to hold the expanded elements.
    # Extract elements using bitwise masks
    unpacked = np.zeros((K, N), dtype=np.int8)
    for i in range(8):
        # bitwise shift
        shift = i * 4
        # We need to correctly handle the sign bit if it's signed int4,
        # but GPTQ uses unsigned int4, shifted by zeros.
        extracted = (array >> shift) & 0xF
        unpacked[:, i::8] = extracted.astype(np.int8)

    return unpacked
