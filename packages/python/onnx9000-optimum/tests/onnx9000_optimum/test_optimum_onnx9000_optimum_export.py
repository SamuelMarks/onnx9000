"""Unit tests for onnx9000.optimum.export and onnx9000_optimum.export."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import onnx9000.optimum
import onnx9000.optimum.architectures as opt_arch
from onnx9000.optimum.export import (
    _progress_bar,
    auto_detect_task,
    create_dummy_inputs,
    export_model,
    get_huggingface_model_files,
    warn_unsupported_ops,
)


def test_optimum_namespace_export_symbols(tmp_path: Path) -> None:
    """Verify onnx9000.optimum.export functions, task auto-detection, and image-classification export."""
    assert onnx9000.optimum.auto_detect_task is auto_detect_task
    assert onnx9000.optimum.export_model is export_model
    assert onnx9000.optimum.get_huggingface_model_files is get_huggingface_model_files
    assert opt_arch.BERTConfig(num_labels=2).options["num_labels"] == 2
    assert (
        auto_detect_task({"architectures": ["BertForSequenceClassification"]})
        == "text-classification"
    )
    assert auto_detect_task({}) == "feature-extraction"
    warn_unsupported_ops()
    assert list(_progress_bar([10, 20])) == [10, 20]
    assert callable(create_dummy_inputs)

    src_dir = tmp_path / "src_model"
    src_dir.mkdir()
    (src_dir / "config.json").write_text('{"architectures": ["ViTForImageClassification"]}')
    out_dir = tmp_path / "out_model"

    mock_transformers = MagicMock()
    mock_torch = MagicMock()
    with (
        patch("onnx9000_optimum.export.get_huggingface_model_files", return_value=str(src_dir)),
        patch.dict(sys.modules, {"transformers": mock_transformers, "torch": mock_torch}),
    ):
        export_model("vit-model", str(out_dir), task="image-classification")
    assert (out_dir / "config.json").exists()
