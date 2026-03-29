"""Module providing onnx2gguf functionality."""

from typing import Any

from onnx9000.core.ir import Graph

from .llama import extract_llama_metadata


def extract_metadata(graph: Graph, arch_override: str = None) -> dict[str, Any]:
    """Extracts metadata."""
    arch = arch_override or infer_architecture(graph)
    infer_architecture(graph)
    if arch_override and arch_override not in [
        "llama",
        "mistral",
        "mixtral",
        "phi2",
        "qwen2",
        "gemma",
        "starcoder",
        "falcon",
        "bloom",
        "stablelm",
        "command-r",
        "bert",
    ]:
        raise ValueError(f"Unsupported strict architecture mapping: {arch_override}")
    if arch == "unknown":
        return {}
    meta = extract_llama_metadata(graph)
    if arch != "llama":
        remapped = {}
        for k, v in meta.items():
            if k.startswith("llama."):
                remapped[k.replace("llama.", f"{arch}.")] = v
            else:
                remapped[k] = v
        meta = remapped
    if arch == "mistral":
        meta["mistral.attention.sliding_window"] = 4096
    elif arch == "gemma":
        meta["gemma.attention.layer_norm_rms_epsilon"] = 1e-06
    elif arch == "mixtral":
        meta["mixtral.expert_count"] = 8
        meta["mixtral.expert_used_count"] = 2
    return meta


def infer_architecture(graph: Graph) -> str:
    """Infers architecture."""
    name = getattr(graph, "name", "").lower()
    if "mistral" in name:
        return "mistral"
    if "mixtral" in name:
        return "mixtral"
    if "phi" in name:
        return "phi2"
    if "qwen" in name:
        return "qwen2"
    if "gemma" in name:
        return "gemma"
    if "starcoder" in name:
        return "starcoder"
    if "falcon" in name:
        return "falcon"
    if "bloom" in name:
        return "bloom"
    if "stablelm" in name:
        return "stablelm"
    if "command-r" in name:
        return "command-r"
    if "bert" in name:
        return "bert"
    text = str(graph.tensors.keys()) + str([n.op_type for n in graph.nodes])
    if "llama" in text.lower():
        return "llama"
    return "unknown"
