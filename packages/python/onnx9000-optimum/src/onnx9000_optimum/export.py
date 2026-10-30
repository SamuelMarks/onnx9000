"""Model export functionality for onnx9000, compatible with HuggingFace Optimum."""

import json
import logging
import os
import shutil
import sys
from collections.abc import Iterable, Iterator, Mapping
from typing import Protocol, TypeAlias, TypeVar

from onnx9000.core.exceptions import Onnx9000Error

logger = logging.getLogger(__name__)

T = TypeVar("T")
ConfigValue: TypeAlias = str | int | float | bool | list[str] | list[int] | None


class TensorSeedProtocol(Protocol):
    """Protocol representing a traced dummy input tensor."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return tensor shape dimensions."""


DummyInputValue: TypeAlias = (
    TensorSeedProtocol | tuple[tuple[TensorSeedProtocol, TensorSeedProtocol], ...]
)


def get_huggingface_model_files(model_id: str, cache_dir: str | None = None) -> str:
    """Download model files from HuggingFace Hub and return the local directory path.

    Args:
        model_id: HuggingFace Hub repository identifier.
        cache_dir: Optional local cache directory path.

    Returns:
        Path to the downloaded local snapshot directory.
    """
    try:
        from huggingface_hub import snapshot_download

        local_dir: str = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            allow_patterns=[
                "*.json",
                "*.bin",
                "*.safetensors",
                "*.model",
                "*.txt",
                "*.py",
            ],
        )
        return local_dir
    except ImportError:
        logger.error("huggingface_hub is not installed. Run `pip install huggingface_hub`.")
        sys.exit(1)


def _progress_bar(iterable: Iterable[T], desc: str = "Progress", unit: str = "it") -> Iterable[T]:
    """Create a progress bar for iterables, with tqdm support if available.

    Args:
        iterable: Input iterable to wrap.
        desc: Description prefix displayed on the progress bar.
        unit: Iteration unit label.

    Returns:
        Wrapped iterable yielding elements from `iterable`.
    """
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, unit=unit)
    except ImportError:
        items = list(iterable)
        total = len(items)

        def generator() -> Iterator[T]:
            """Yield items while printing a simple progress update to stdout."""
            cr = chr(13)
            nl = chr(10)
            for i, item in enumerate(items):
                sys.stdout.write(f"{cr}{desc}: {i}/{total} [{int(i / total * 100)}%] ")
                sys.stdout.flush()
                yield item
            sys.stdout.write(f"{cr}{desc}: {total}/{total} [100%]{nl}")

        return generator()


def auto_detect_task(config: Mapping[str, ConfigValue]) -> str:
    """Auto-detect the Optimum export task from `config.json` architectures.

    Args:
        config: Parsed `config.json` dictionary.

    Returns:
        Canonical task identifier string.
    """
    architectures = config.get("architectures", [])
    if not isinstance(architectures, list) or not architectures:
        return "feature-extraction"
    arch = str(architectures[0])
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


def warn_unsupported_ops() -> None:
    """Log a warning if unsupported PyTorch ops are encountered during export."""
    logger.warning(
        "Unsupported PyTorch ops found during export. Fallback to CPU recommended if WebGPU fails."
    )


def create_dummy_inputs(
    config: Mapping[str, ConfigValue], task: str, use_past: bool = False
) -> dict[str, DummyInputValue]:
    """Create dummy inputs for ONNX JIT tracing.

    Args:
        config: Model configuration mapping.
        task: Target task name.
        use_past: Whether to include `past_key_values` inputs for KV caching.

    Returns:
        Dictionary of input names to dummy torch tensors.
    """
    import torch

    batch_size = 1
    seq_length = 8
    inputs: dict[str, DummyInputValue] = {}

    vocab_size_raw = config.get("vocab_size", 1000)
    vocab_size = int(vocab_size_raw) if isinstance(vocab_size_raw, (int, float)) else 1000

    if task in ["text-generation", "fill-mask", "text-classification"]:
        inputs["input_ids"] = torch.randint(0, vocab_size, (batch_size, seq_length))
        inputs["attention_mask"] = torch.ones((batch_size, seq_length), dtype=torch.int64)
        if use_past:
            num_heads_raw = config.get("num_attention_heads", 12)
            hidden_size_raw = config.get("hidden_size", 768)
            num_layers_raw = config.get("num_hidden_layers", 12)
            num_heads = int(num_heads_raw) if isinstance(num_heads_raw, (int, float)) else 12
            hidden_size = int(hidden_size_raw) if isinstance(hidden_size_raw, (int, float)) else 768
            num_layers = int(num_layers_raw) if isinstance(num_layers_raw, (int, float)) else 12
            head_dim = hidden_size // num_heads
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
        inputs["input_ids"] = torch.randint(0, 1000, (batch_size, seq_length))

    return inputs


def export_model(
    model_id: str,
    output_dir: str,
    task: str | None = None,
    opset: int | None = 14,
    device: str = "cpu",
    cache_dir: str | None = None,
    split: bool = False,
) -> None:
    """Trace and export a HuggingFace model to ONNX format.

    Args:
        model_id: HuggingFace Hub model identifier.
        output_dir: Destination directory for the exported ONNX model and metadata.
        task: Optional task name (auto-detected if `None`).
        opset: Target ONNX opset version.
        device: Execution device string.
        cache_dir: Optional HuggingFace cache directory.
        split: Whether to split the exported model into multiple components.
    """
    print(f"Exporting model {model_id}...")
    local_path = get_huggingface_model_files(model_id, cache_dir)

    config_path = os.path.join(local_path, "config.json")
    if not os.path.exists(config_path):
        logger.error(f"config.json not found in {local_path}")
        sys.exit(1)

    with open(config_path) as f:
        config: dict[str, ConfigValue] = json.load(f)

    if task is None:
        task = auto_detect_task(config)
        print(f"Auto-detected task: {task}")

    import torch

    try:
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
        )

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

    use_past = bool(getattr(model.config, "use_cache", False)) and task == "text-generation"

    dummy_inputs = create_dummy_inputs(config, task, use_past)

    dynamic_axes: dict[str, dict[int, str]] = {}
    if "input_ids" in dummy_inputs:
        dynamic_axes["input_ids"] = {0: "batch_size", 1: "sequence_length"}
    if "attention_mask" in dummy_inputs:
        dynamic_axes["attention_mask"] = {0: "batch_size", 1: "sequence_length"}

    output_names = ["logits"]
    if use_past:
        dynamic_axes["logits"] = {0: "batch_size", 1: "sequence_length"}
        num_layers_raw = config.get("num_hidden_layers", 12)
        num_layers = int(num_layers_raw) if isinstance(num_layers_raw, (int, float)) else 12
        for i in range(num_layers):
            dynamic_axes[f"past_key_values_{i}_key"] = {
                0: "batch_size",
                2: "sequence_length",
            }
            dynamic_axes[f"past_key_values_{i}_value"] = {
                0: "batch_size",
                2: "sequence_length",
            }
            output_names.extend([f"present_{i}_key", f"present_{i}_value"])

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model.onnx")

    print("Tracing and exporting to ONNX...")
    for _ in _progress_bar([1], desc="Exporting"):
        with torch.no_grad():
            try:
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
            except (
                Onnx9000Error,
                OSError,
                ValueError,
                TypeError,
                RuntimeError,
                KeyError,
                IndexError,
                AttributeError,
                ImportError,
                LookupError,
                ArithmeticError,
                BufferError,
                AssertionError,
                StopIteration,
            ) as e:
                warn_unsupported_ops()
                logger.error(f"Export failed: {e}")
                sys.exit(1)

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
