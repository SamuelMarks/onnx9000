from .naming import rename_tensor
from .tokenizer import extract_tokenizer_metadata
import re
from typing import BinaryIO, Any, Optional
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from .builder import GGUFWriter, GGUFValueType, GGUFTensorType


def get_gguf_type(dtype: int) -> GGUFTensorType:
    if dtype == DType.FLOAT32.value:
        return GGUFTensorType.F32
    if dtype == DType.FLOAT16.value:
        return GGUFTensorType.F16
    return GGUFTensorType.F32


def infer_architecture(graph: Graph) -> str:
    # Very basic heuristic for Phase 2, to be expanded in Phase 3/4
    # Check tensor names or node types
    text = str(graph.name) + str(graph.tensors.keys()) + str([n.op_type for n in graph.nodes])
    if "llama" in text.lower() or "Llama" in text:
        return "llama"
    return "unknown"


def sanitize_doc_string(doc: str) -> str:
    if not doc:
        return ""
    # Strip non-utf8 or weird controls if needed
    return doc.strip()


def compile_gguf(graph: Graph, out_f: BinaryIO, kv_overrides: Optional[dict[str, Any]] = None):
    writer = GGUFWriter(out_f)
    kv_overrides = kv_overrides or {}

    arch = infer_architecture(graph)

    # Phase 2 defaults
    general_kvs = {
        "general.architecture": arch,
        "general.name": graph.name if graph.name else "model",
        "general.author": getattr(graph, "producer_name", "onnx9000") or "onnx9000",
        "general.version": getattr(graph, "model_version", 1) or 1,
        "general.quantization_version": 2,
        "general.alignment": 32,
        "general.file_type": "mostly_f32",
    }

    doc_str = getattr(graph, "doc_string", "")
    if doc_str:
        general_kvs["general.description"] = sanitize_doc_string(doc_str)

    for k, v in kv_overrides.items():
        if k.startswith("general."):
            general_kvs[k] = v

    for k, v in general_kvs.items():
        if isinstance(v, bool):
            writer.add_bool(k, v)
        elif isinstance(v, str):
            writer.add_string(k, v)
        elif isinstance(v, int):
            writer.add_uint32(k, v)
        elif isinstance(v, float):
            writer.add_float32(k, v)

    # Llama specifics (if it is llama)
    if general_kvs["general.architecture"] == "llama":
        writer.add_uint32("llama.context_length", 2048)
        writer.add_uint32("llama.embedding_length", 4096)
        writer.add_uint32("llama.block_count", 32)
        writer.add_uint32("llama.feed_forward_length", 11008)
        writer.add_uint32("llama.attention.head_count", 32)
        writer.add_uint32("llama.attention.head_count_kv", 32)
        writer.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)

    # Phase 5: Tokenizer
    tok_meta = extract_tokenizer_metadata(
        kv_overrides.get("tokenizer.json"), general_kvs.get("llama.vocab_size", 0)
    )
    for k, v in tok_meta.items():
        if k not in kv_overrides:
            if isinstance(v, str):
                writer.add_string(k, v)
            elif isinstance(v, bool):
                writer.add_bool(k, v)
            elif isinstance(v, int):
                writer.add_uint32(k, v)
            elif isinstance(v, float):
                writer.add_float32(k, v)
            elif isinstance(v, list):
                if v and isinstance(v[0], str):
                    writer.add_array(k, v, GGUFValueType.STRING)
                elif v and isinstance(v[0], float):
                    writer.add_array(k, v, GGUFValueType.FLOAT32)
                elif v and isinstance(v[0], int):
                    writer.add_array(k, v, GGUFValueType.INT32)

    for k, v in kv_overrides.items():
        if not k.startswith("general."):
            if isinstance(v, bool):
                writer.add_bool(k, v)
            elif isinstance(v, str):
                writer.add_string(k, v)
            elif isinstance(v, int):
                writer.add_uint32(k, v)
            elif isinstance(v, float):
                writer.add_float32(k, v)

    # Phase 6 & 7: Tensors
    for init_name in graph.initializers:
        if init_name in graph.tensors:
            init = graph.tensors[init_name]
            if isinstance(init, Tensor):
                try:
                    gguf_name = rename_tensor(
                        init_name, kv_overrides.get("tensorNameOverrides", {})
                    )
                except ValueError:
                    continue
                shape = [int(s) if not isinstance(s, str) else 1 for s in init.shape]
                writer.add_tensor_info(gguf_name, list(reversed(shape)), get_gguf_type(init.dtype))

    writer.write_header_to_file()

    for init_name in graph.initializers:
        if init_name in graph.tensors:
            init = graph.tensors[init_name]
            if isinstance(init, Tensor) and init.data is not None:
                writer.write_tensor_data(init.data)
