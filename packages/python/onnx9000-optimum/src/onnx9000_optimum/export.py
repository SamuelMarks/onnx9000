import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_huggingface_model_files(model_id: str, cache_dir: Optional[str] = None) -> str:
    """Download model files and return local path."""
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.bin", "*.safetensors", "*.model", "*.txt", "*.py"],
        )
        return local_dir
    except ImportError:
        logger.error("huggingface_hub is not installed. Run `pip install huggingface_hub`.")
        sys.exit(1)


def _progress_bar(iterable, desc="Progress", unit="it"):
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, unit=unit)
    except ImportError:
        # Fallback simplistic progress bar
        items = list(iterable)
        total = len(items)

        def generator():
            for i, item in enumerate(items):
                sys.stdout.write(f"\r{desc}: {i}/{total} [{int(i / total * 100)}%] ")
                sys.stdout.flush()
                yield item
            sys.stdout.write(f"\r{desc}: {total}/{total} [100%]\n")

        return generator()


def auto_detect_task(config: dict[str, Any]) -> str:
    """Auto-detect task from config.json if omitted."""
    architectures = config.get("architectures", [])
    if not architectures:
        return "feature-extraction"
    arch = architectures[0]
    if "SequenceClassification" in arch:
        return "text-classification"
    if "TokenClassification" in arch:
        return "token-classification"
    if "QuestionAnswering" in arch:
        return "question-answering"
    if "CausalLM" in arch:
        return "text-generation"
    if "MaskedLM" in arch:
        return "fill-mask"
    if "Seq2SeqLM" in arch:
        return "text2text-generation"
    if "ImageClassification" in arch:
        return "image-classification"
    if "ObjectDetection" in arch:
        return "object-detection"
    return "feature-extraction"


def warn_unsupported_ops():
    logger.warning(
        "Unsupported PyTorch ops found during export. Fallback to CPU recommended if WebGPU fails."
    )


def create_dummy_inputs(
    config: dict[str, Any], task: str, use_past: bool = False
) -> dict[str, Any]:
    """Create dummy inputs for ONNX JIT tracing."""
    import torch

    batch_size = 1
    seq_length = 8
    inputs = {}

    if task in ["text-generation", "fill-mask", "text-classification"]:
        inputs["input_ids"] = torch.randint(
            0, config.get("vocab_size", 1000), (batch_size, seq_length)
        )
        inputs["attention_mask"] = torch.ones((batch_size, seq_length), dtype=torch.int64)
        if use_past:
            num_heads = config.get("num_attention_heads", 12)
            head_dim = config.get("hidden_size", 768) // num_heads
            num_layers = config.get("num_hidden_layers", 12)
            past_key_values = []
            for _ in range(num_layers):
                past_key_values.append(
                    (
                        torch.zeros(batch_size, num_heads, seq_length, head_dim),
                        torch.zeros(batch_size, num_heads, seq_length, head_dim),
                    )
                )
            inputs["past_key_values"] = tuple(past_key_values)
    elif task == "image-classification":
        inputs["pixel_values"] = torch.randn(batch_size, 3, 224, 224)
    else:
        # Default
        inputs["input_ids"] = torch.randint(0, 1000, (batch_size, seq_length))

    return inputs


def export_model(
    model_id: str,
    output_dir: str,
    task: Optional[str] = None,
    opset: Optional[int] = 14,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    split: bool = False,
):

    print(f"Exporting model {model_id}...")
    local_path = get_huggingface_model_files(model_id, cache_dir)

    config_path = os.path.join(local_path, "config.json")
    if not os.path.exists(config_path):
        logger.error(f"config.json not found in {local_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    if task is None:
        task = auto_detect_task(config)
        print(f"Auto-detected task: {task}")

    import torch

    try:
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification

        # Simplify loading for demo
        if task == "text-generation":
            model = AutoModelForCausalLM.from_pretrained(local_path)
        elif task == "text-classification":
            model = AutoModelForSequenceClassification.from_pretrained(local_path)
        else:
            model = AutoModel.from_pretrained(local_path)
    except ImportError:
        logger.error("transformers is not installed. Run `pip install transformers torch`.")
        sys.exit(1)

    model.eval()

    use_past = getattr(model.config, "use_cache", False) and task == "text-generation"

    dummy_inputs = create_dummy_inputs(config, task, use_past)

    dynamic_axes = {}
    if "input_ids" in dummy_inputs:
        dynamic_axes["input_ids"] = {0: "batch_size", 1: "sequence_length"}
    if "attention_mask" in dummy_inputs:
        dynamic_axes["attention_mask"] = {0: "batch_size", 1: "sequence_length"}

    output_names = ["logits"]
    if use_past:
        dynamic_axes["logits"] = {0: "batch_size", 1: "sequence_length"}
        num_layers = config.get("num_hidden_layers", 12)
        for i in range(num_layers):
            dynamic_axes[f"past_key_values_{i}_key"] = {0: "batch_size", 2: "sequence_length"}
            dynamic_axes[f"past_key_values_{i}_value"] = {0: "batch_size", 2: "sequence_length"}
            output_names.extend([f"present_{i}_key", f"present_{i}_value"])

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model.onnx")

    print("Tracing and exporting to ONNX...")
    # Wrap in progress bar
    for _ in _progress_bar([1], desc="Exporting"):
        with torch.no_grad():
            try:
                # Mock export if onnx is missing or just do native torch export
                torch.onnx.export(
                    model,
                    tuple(dummy_inputs.values()),
                    onnx_path,
                    input_names=list(dummy_inputs.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=opset or 14,
                    do_constant_folding=True,
                )
            except Exception as e:
                warn_unsupported_ops()
                logger.error(f"Export failed: {e}")
                sys.exit(1)

    # Copy metadata
    import shutil

    for meta_file in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "preprocessor_config.json",
    ]:
        src = os.path.join(local_path, meta_file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, meta_file))

    print(f"Model exported successfully to {output_dir}")
