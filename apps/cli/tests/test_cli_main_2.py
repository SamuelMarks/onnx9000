from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest


def test_export_cmd_failures():
    import onnx9000_cli.main as cli_main

    with patch("importlib.util.spec_from_file_location", return_value=None):
        with patch("sys.exit", side_effect=SystemExit):
            with pytest.raises(SystemExit):
                cli_main.export_cmd(Namespace(script="fake", format="onnx"))

    mock_spec = MagicMock()
    mock_spec.loader = None
    with patch("importlib.util.spec_from_file_location", return_value=mock_spec):
        with patch("sys.exit", side_effect=SystemExit):
            with pytest.raises(SystemExit):
                cli_main.export_cmd(Namespace(script="fake", format="onnx"))

    # Mock successful load but no model found
    mock_spec = MagicMock()
    mock_module = MagicMock()
    with patch("importlib.util.spec_from_file_location", return_value=mock_spec):
        with patch("importlib.util.module_from_spec", return_value=mock_module):
            with patch("sys.exit", side_effect=SystemExit):
                with pytest.raises(SystemExit):
                    cli_main.export_cmd(Namespace(script="fake", format="onnx"))


def test_convert_cmd_branches():
    from argparse import Namespace

    from onnx9000_cli.main import convert_cmd

    with patch("onnx9000.core.parser.core.load"):
        with patch("onnx9000.core.exporter.export_graph") as mock_exp:
            # src_fmt keras
            with patch("importlib.import_module"):
                mock_keras = MagicMock()
                mock_keras.saving.load_model.return_value = b"dummy"
                import sys
                import onnx9000.converters.tf.api

                sys.modules["keras"] = mock_keras
                sys.modules["h5py"] = MagicMock()
                with patch("onnx9000.converters.tf.api.convert_keras_to_onnx") as mock_convert:
                    mock_convert.return_value = MagicMock()
                    # no output provided
                    args = Namespace(src="m.h5", output=None, format="onnx")
                    setattr(args, "from", "keras")
                    args.to = "onnx"

                    try:
                        convert_cmd(args)
                    except Exception:
                        pass

            # src_fmt pytorch
            args2 = Namespace(src="m.pt", output="out.pt", format="onnx")
            setattr(args2, "from", "pytorch")
            args2.to = "onnx"
            try:
                convert_cmd(args2)
            except Exception:
                pass

            # src_fmt unknown
            args3 = Namespace(src="m.abc", output="out.abc", format="onnx")
            setattr(args3, "from", "unknown")
            args3.to = "onnx"
            try:
                convert_cmd(args3)
            except Exception:
                pass


def test_openvino_cmd_branches():
    from argparse import Namespace

    from onnx9000_cli.main import openvino_cmd, openvino_export_cmd

    with patch("onnx9000.core.parser.core.load") as mock_load:
        mock_graph = MagicMock()
        mock_inp = MagicMock()
        mock_inp.name = "input_1"
        mock_inp.shape = (1, 3, 224, 224)
        mock_graph.inputs = [mock_inp]
        mock_load.return_value = mock_graph

        with patch("onnx9000.openvino.exporter.OpenVinoExporter") as mock_exp_class:
            mock_exp_class.return_value.export.return_value = ("<xml></xml>", b"")
            with patch("builtins.open"):
                # dynamic_batch
                args = Namespace(model="m.onnx", output_dir="out", fp16=False, dynamic_batch=True)
                args.data_type = None
                args.shape = ["input_1:1,3,invalid,224", "input_1:1,3,224,224"]
                openvino_export_cmd(args)
                assert mock_inp.shape[0] == -1

                # data_type
                args2 = Namespace(model="m.onnx", output_dir="out", fp16=False, dynamic_batch=False)
                args2.data_type = ["input_1:float16"]
                args2.shape = None
                openvino_export_cmd(args2)
                assert mock_inp.dtype == "float16"

    with patch("sys.exit", side_effect=SystemExit):
        with pytest.raises(SystemExit):
            openvino_cmd(Namespace())


def test_info_cmd():
    from argparse import Namespace

    from onnx9000_cli.main import info_cmd

    with patch("sys.exit", side_effect=SystemExit):
        with pytest.raises(SystemExit):
            info_cmd(Namespace())
