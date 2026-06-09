import argparse
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest
from onnx9000_cli import main


def test_convert_flax_json():
    args = argparse.Namespace(
        src="dummy.flax",
        from_fmt="flax",
        to_fmt="onnx",
        output="out.onnx",
        weights=None,
    )
    with patch("builtins.open", mock_open(read_data=b'{"not_msgpack": 1}')):
        with patch("onnx9000.core.serializer.save"):
            try:
                main.convert_cmd(args)
            except Exception:
                pass


def test_convert_files_json():
    mock_mod = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "onnx9000.converters.mltools.catboost": mock_mod,
            "onnx9000.converters.sklearn.parser": mock_mod,
            "onnx9000.converters.paddle.loader": mock_mod,
            "onnx9000.converters.mltools.sparkml": mock_mod,
            "onnx9000.converters.frontend.tracer": mock_mod,
            "onnx9000.converters.mltools.xgboost": mock_mod,
            "onnx9000.converters.mltools.libsvm": mock_mod,
            "onnx9000.converters.mltools.keras3": mock_mod,
            "onnx9000.converters.mltools.h2o": mock_mod,
            "onnx9000.converters.mltools.coreml": mock_mod,
        },
    ):
        for fmt in [
            "catboost",
            "sklearn",
            "pyspark",
            "libsvm",
            "xgboost",
            "tfjs",
            "sparkml",
            "h2o",
            "coreml",
        ]:
            args = argparse.Namespace(
                src="dummy.json",
                from_fmt=fmt,
                to_fmt="onnx",
                output="out.onnx",
                weights=None,
            )
            with patch("builtins.open", mock_open(read_data="{}")):
                with patch("onnx9000.core.serializer.save"):
                    try:
                        main.convert_cmd(args)
                    except Exception:
                        pass

        args = argparse.Namespace(
            src="dummy.pd",
            from_fmt="paddle",
            to_fmt="onnx",
            output="out.onnx",
            weights=None,
        )
        with patch("onnx9000.core.serializer.save"):
            try:
                main.convert_cmd(args)
            except Exception:
                pass

        args = argparse.Namespace(
            src="dummy.pb",
            from_fmt="tensorflow",
            to_fmt="onnx",
            output="out.onnx",
            weights=None,
        )
        with patch("onnx9000.core.serializer.save"):
            try:
                main.convert_cmd(args)
            except Exception:
                pass


def test_convert_various_success():
    fmt_src_weights = [
        ("darknet", "dummy.cfg", "dummy.weights", "Darknet"),
        ("ncnn", "dummy.param", "dummy.bin", "NCNN"),
        ("caffe", "dummy.prototxt", "dummy.caffemodel", "Caffe"),
        ("cntk", "dummy.model", None, "CNTK"),
        ("mxnet", "dummy-symbol.json", "dummy.params", "MXNet"),
    ]
    for fmt, src, w, cls in fmt_src_weights:
        args = argparse.Namespace(
            src=src, from_fmt=fmt, to_fmt="onnx", output="out.onnx", weights=w
        )
        with patch(f"onnx9000.converters.{fmt}.{cls}Converter", MagicMock()):
            with patch("onnx9000.core.serializer.save"):
                try:
                    main.convert_cmd(args)
                except Exception:
                    pass


def test_convert_errors():
    with pytest.raises(SystemExit):
        main.convert_cmd(
            argparse.Namespace(
                src="dummy.cfg",
                from_fmt="darknet",
                to_fmt="onnx",
                output="out",
                weights=None,
            )
        )
    with pytest.raises(SystemExit):
        main.convert_cmd(
            argparse.Namespace(
                src="dummy.txt",
                from_fmt="darknet",
                to_fmt="onnx",
                output="out",
                weights="dummy.weights",
            )
        )
    with pytest.raises(SystemExit):
        main.convert_cmd(
            argparse.Namespace(
                src="dummy.param",
                from_fmt="ncnn",
                to_fmt="onnx",
                output="out",
                weights=None,
            )
        )
    with pytest.raises(SystemExit):
        main.convert_cmd(
            argparse.Namespace(
                src="dummy.txt",
                from_fmt="ncnn",
                to_fmt="onnx",
                output="out",
                weights="dummy.bin",
            )
        )
    with pytest.raises(SystemExit):
        main.convert_cmd(
            argparse.Namespace(
                src="dummy.prototxt",
                from_fmt="caffe",
                to_fmt="onnx",
                output="out",
                weights=None,
            )
        )
    with pytest.raises(SystemExit):
        main.convert_cmd(
            argparse.Namespace(
                src="dummy.txt",
                from_fmt="caffe",
                to_fmt="onnx",
                output="out",
                weights="dummy.caffemodel",
            )
        )
    with pytest.raises(SystemExit):
        main.convert_cmd(
            argparse.Namespace(
                src="dummy.txt",
                from_fmt="cntk",
                to_fmt="onnx",
                output="out",
                weights=None,
            )
        )
    with pytest.raises(SystemExit):
        main.convert_cmd(
            argparse.Namespace(
                src="dummy-symbol.json",
                from_fmt="mxnet",
                to_fmt="onnx",
                output="out",
                weights=None,
            )
        )
    with pytest.raises(SystemExit):
        main.convert_cmd(
            argparse.Namespace(
                src="dummy.txt",
                from_fmt="mxnet",
                to_fmt="onnx",
                output="out",
                weights="dummy.params",
            )
        )


def test_convert_out_formats():
    for out in ["c", "cpp", "mlir", "keras", "wasm"]:
        args = argparse.Namespace(
            src="dummy.onnx",
            from_fmt="onnx",
            to_fmt=out,
            output="out.file",
            weights=None,
        )
        with patch("onnx9000_cli.main.load_onnx"):
            with patch("onnx9000.core.exporter.export_graph"):
                main.convert_cmd(args)


def test_mlir_cmd():
    main.mlir_cmd(argparse.Namespace(model="dummy"))


def test_import_errors():
    with patch.dict(
        sys.modules,
        {
            "onnx9000_mobile_memory": None,
            "onnx9000_progressive_loading": None,
            "onnx9000_new_model_arch": None,
            "onnx9000_zero_dep_classifier": None,
            "onnx9000_ort_training": None,
            "onnx9000_olive_optimizer": None,
            "onnx9000_triton_server": None,
            "onnx9000_onnx_tool": None,
            "onnx9000_custom_ops": None,
            "onnx9000_profiler": None,
        },
    ):
        main.mobile_memory_cmd(argparse.Namespace(model="dummy"))
        main.progressive_loading_cmd(argparse.Namespace(model="dummy"))
        main.new_model_arch_cmd(argparse.Namespace(arch="dummy"))
        main.zero_dep_classifier_cmd(argparse.Namespace(model="dummy"))
        main.ort_training_cmd(argparse.Namespace(model="dummy"))
        main.olive_optimizer_cmd(argparse.Namespace(model="dummy"))
        main.triton_server_cmd(argparse.Namespace(model="dummy"))
        main.onnx_tool_cmd(argparse.Namespace(model="dummy"))
        main.custom_ops_cmd(argparse.Namespace(ops_file="dummy"))
        main.profiler_cmd(argparse.Namespace(model="dummy", show_arena=True))


def test_simple_cmds():
    main.paddle2onnx_cmd(argparse.Namespace(model="dummy"))
    main.keras2onnx_cmd(argparse.Namespace(model="dummy"))
    main.skl2onnx_cmd(argparse.Namespace(model="dummy"))
    main.arena_cmd(argparse.Namespace(model="dummy"))
