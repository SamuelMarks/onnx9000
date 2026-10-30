"""Unit tests for onnx9000.optimum.quantize and onnx9000_optimum.quantize."""

from unittest.mock import MagicMock, patch

from onnx9000.optimum.quantize import (
    CalibrationDataReader,
    Quantizer,
    awq_quantize,
    blockwise_quantize,
    export_calibration_data,
    quantize_model,
    smooth_quant,
)


def test_quantize_namespace_functions() -> None:
    """Verify CalibrationDataReader, Quantizer, and quantization helpers via onnx9000.optimum.quantize."""
    reader = CalibrationDataReader("custom_data")
    assert reader.dataset_name == "custom_data"
    assert list(reader) == [{"input_ids": [0]}]
    assert export_calibration_data(reader, "out.bin") is False
    assert blockwise_quantize(None) is False
    assert awq_quantize(None) is False
    assert smooth_quant(None) is False

    with patch("onnx9000_optimum.quantize.quantize_model") as mock_qm:
        Quantizer.quantize("model.onnx", {"method": "gptq", "gptq_bits": 8, "gptq_group_size": 64})
        mock_qm.assert_called_once_with(
            "model.onnx", method="gptq", gptq_bits=8, gptq_group_size=64
        )

    with (
        patch("onnx9000.core.parser.core.load", return_value=MagicMock()),
        patch("onnx9000.core.serializer.save"),
    ):
        quantize_model("model.onnx", method="custom_mode")
