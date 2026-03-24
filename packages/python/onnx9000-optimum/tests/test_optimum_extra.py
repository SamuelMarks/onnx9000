import pytest
import sys
from unittest.mock import patch, MagicMock
import onnx9000_optimum
import onnx9000_optimum.export
import onnx9000_optimum.optimize
import onnx9000_optimum.quantize
import onnx9000_optimum.architectures


@patch("onnx9000_optimum.export.get_huggingface_model_files")
@patch("os.path.exists")
@patch("builtins.open")
@patch("json.load")
@patch("onnx9000_optimum.export._progress_bar")
@patch("os.makedirs")
@patch("shutil.copy")
def test_export_model_flow(
    mock_copy, mock_makedirs, mock_pbar, mock_json_load, mock_open, mock_exists, mock_get_hf
):
    from onnx9000_optimum.export import export_model

    mock_get_hf.return_value = "/tmp/fake_model"
    mock_exists.return_value = True
    mock_json_load.return_value = {"architectures": ["CausalLM"]}
    mock_pbar.return_value = [1]

    # Mock torch and transformers
    import sys

    mock_torch = MagicMock()
    mock_torch.no_grad = MagicMock()
    mock_transformers = MagicMock()

    with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
        export_model("model_id", "/tmp/out")
        mock_get_hf.assert_called_once()
        mock_makedirs.assert_called()


@patch("onnx9000.core.parser.core.load")
@patch("onnx9000.core.serializer.save")
@patch("onnx9000.optimizer.simplifier.api.simplify")
@patch("os.path.getsize")
def test_optimize_model_flow(mock_getsize, mock_simplify, mock_save, mock_load):
    from onnx9000_optimum.optimize import optimize_model

    mock_getsize.side_effect = [1024, 512]

    # Test level O1
    optimize_model("test.onnx", level="O1")

    # Test level O4
    optimize_model("test.onnx", level="O4")


@patch("onnx9000.core.parser.core.load")
@patch("onnx9000.core.serializer.save")
def test_quantize_model_flow(mock_save, mock_load):
    from onnx9000_optimum.quantize import quantize_model

    quantize_model("test.onnx", method="dynamic")
    quantize_model("test.onnx", method="static")
    quantize_model("test.onnx", method="gptq")


def test_missing_imports():
    from onnx9000_optimum.export import (
        export_model,
        get_huggingface_model_files,
        _progress_bar,
        warn_unsupported_ops,
    )
    from onnx9000_optimum.optimize import optimize_model
    from onnx9000_optimum.quantize import quantize_model
    import sys

    with patch("builtins.__import__", side_effect=ImportError):
        with pytest.raises(SystemExit):
            get_huggingface_model_files("test")

        with pytest.raises(SystemExit):
            optimize_model("test")

        with pytest.raises(SystemExit):
            quantize_model("test")

        warn_unsupported_ops()

    res = _progress_bar([1, 2, 3])
    list(res)


def test_export_unsupported():
    from onnx9000_optimum.export import export_model

    with patch("onnx9000_optimum.export.get_huggingface_model_files") as mock_get_hf:
        mock_get_hf.return_value = "/tmp/fake_model"
        with patch("os.path.exists", return_value=False):
            with pytest.raises(SystemExit):
                export_model("model_id", "/tmp/out")


def test_auto_detect_others():
    from onnx9000_optimum.export import auto_detect_task

    assert auto_detect_task({"architectures": ["TokenClassification"]}) == "token-classification"
    assert auto_detect_task({"architectures": ["QuestionAnswering"]}) == "question-answering"
    assert auto_detect_task({"architectures": ["MaskedLM"]}) == "fill-mask"
    assert auto_detect_task({"architectures": ["Seq2SeqLM"]}) == "text2text-generation"
    assert auto_detect_task({"architectures": ["ImageClassification"]}) == "image-classification"
    assert auto_detect_task({"architectures": ["ObjectDetection"]}) == "object-detection"


def test_dummy_inputs():
    from onnx9000_optimum.export import create_dummy_inputs
    import sys
    from unittest.mock import MagicMock

    mock_torch = MagicMock()
    with patch.dict(sys.modules, {"torch": mock_torch}):
        create_dummy_inputs({"vocab_size": 10}, "text-generation", use_past=True)
        create_dummy_inputs({"vocab_size": 10}, "image-classification", use_past=False)
        create_dummy_inputs({"vocab_size": 10}, "other", use_past=False)


def test_export_other_tasks():
    from onnx9000_optimum.export import export_model
    import sys
    from unittest.mock import MagicMock

    mock_transformers = MagicMock()
    mock_transformers.AutoModelForSequenceClassification.from_pretrained.return_value = MagicMock()
    mock_transformers.AutoModel.from_pretrained.return_value = MagicMock()
    mock_torch = MagicMock()

    with (
        patch("onnx9000_optimum.export.get_huggingface_model_files", return_value="/tmp"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open"),
        patch("json.load", return_value={"architectures": ["SequenceClassification"]}),
        patch("onnx9000_optimum.export._progress_bar", return_value=[1]),
        patch("os.makedirs"),
        patch("shutil.copy"),
        patch.dict(sys.modules, {"transformers": mock_transformers, "torch": mock_torch}),
    ):
        export_model("model_id", "/tmp/out", task="text-classification")
        export_model("model_id", "/tmp/out", task="other")


def test_export_exception():
    from onnx9000_optimum.export import export_model
    import sys

    mock_torch = MagicMock()
    mock_torch.no_grad = MagicMock()
    mock_torch.onnx.export.side_effect = Exception("mocked err")

    mock_transformers = MagicMock()
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()

    with (
        patch("onnx9000_optimum.export.get_huggingface_model_files", return_value="/tmp"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open"),
        patch("json.load", return_value={"architectures": ["SequenceClassification"]}),
        patch("onnx9000_optimum.export._progress_bar", return_value=[1]),
        patch("os.makedirs"),
        patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}),
    ):
        with pytest.raises(SystemExit):
            export_model("model_id", "/tmp/out")


def test_quantizer_methods():
    from onnx9000_optimum.quantize import (
        export_calibration_data,
        blockwise_quantize,
        awq_quantize,
        smooth_quant,
        CalibrationDataReader,
    )

    reader = CalibrationDataReader()
    list(reader)
    export_calibration_data(reader, "out")
    blockwise_quantize(None)
    awq_quantize(None)
    smooth_quant(None)

    from onnx9000_optimum.quantize import Quantizer

    with patch("onnx9000_optimum.quantize.quantize_model"):
        Quantizer.quantize("model", {})


def test_export_tqdm_and_hub_success():
    import sys
    from unittest.mock import MagicMock

    hf_mock = MagicMock()
    hf_mock.snapshot_download = MagicMock(return_value="/tmp/test")

    tqdm_mock = MagicMock()
    tqdm_mock.tqdm = MagicMock(return_value=[1, 2, 3])

    with patch.dict(sys.modules, {"huggingface_hub": hf_mock, "tqdm": tqdm_mock}):
        from onnx9000_optimum.export import (
            get_huggingface_model_files,
            _progress_bar,
            auto_detect_task,
        )

        assert get_huggingface_model_files("test") == "/tmp/test"
        assert list(_progress_bar([1, 2, 3])) == [1, 2, 3]
        assert auto_detect_task({"architectures": []}) == "feature-extraction"


def test_export_transformers_error():
    from onnx9000_optimum.export import export_model
    import sys
    from unittest.mock import MagicMock

    with patch.dict(sys.modules, {"transformers": None, "torch": MagicMock()}):
        with (
            patch("onnx9000_optimum.export.get_huggingface_model_files", return_value="/tmp/test"),
            patch("os.path.exists", return_value=True),
            patch("builtins.open"),
            patch("json.load", return_value={}),
        ):
            with pytest.raises(SystemExit):
                export_model("test", "/tmp")


def test_optimize_import_error():
    from onnx9000_optimum.optimize import optimize_model
    import sys

    with patch.dict("sys.modules", {"onnx9000.optimizer.simplifier.api": None}):
        with patch("onnx9000.core.parser.core.load", return_value=MagicMock()):
            with pytest.raises(SystemExit):
                optimize_model("test")


def test_auto_detect_unknown():
    from onnx9000_optimum.export import auto_detect_task

    assert auto_detect_task({"architectures": ["Unknown"]}) == "feature-extraction"
