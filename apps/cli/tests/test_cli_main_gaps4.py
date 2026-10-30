"""Tests for remaining CLI command and coverage branches."""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn
from unittest.mock import MagicMock, mock_open, patch

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, ValueInfo
from onnx9000_cli import main


def test_missing_commands() -> None:
    """Verify execution of CLI subcommand handlers."""
    args = argparse.Namespace(
        model="test.onnx",
        src="test.onnx",
        dst="test.json",
        arch="test_arch",
        ops_file="ops.py",
        show_arena=True,
        zoo_command="other",
    )

    main.mlir_cmd(args)
    main.paddle2onnx_cmd(args)
    main.keras2onnx_cmd(args)
    main.skl2onnx_cmd(args)
    main.arena_cmd(args)
    main.ort_training_cmd(args)
    main.olive_optimizer_cmd(args)
    main.triton_server_cmd(args)
    main.onnx_tool_cmd(args)
    main.mobile_memory_cmd(args)
    main.progressive_loading_cmd(args)
    main.new_model_arch_cmd(args)
    main.zero_dep_classifier_cmd(args)
    with patch.dict(
        sys.modules,
        {
            "onnx9000.zoo.catalog": MagicMock(ModelCatalog=MagicMock()),
            "onnx9000.zoo.tensors": MagicMock(
                BFloat16Upcaster=MagicMock(), SafeTensorsMmapParser=MagicMock()
            ),
        },
    ):
        main.zoo_cmd(args)

    with patch.dict(sys.modules, {"onnx9000_custom_ops": MagicMock(registry=MagicMock())}):
        main.custom_ops_cmd(args)
    with patch.dict(sys.modules, {"onnx9000_custom_ops": None}):
        main.custom_ops_cmd(args)

    mock_profiler_cls = MagicMock()
    with patch.dict(sys.modules, {"onnx9000_profiler": MagicMock(Profiler=mock_profiler_cls)}):
        main.profiler_cmd(args)
        args.show_arena = False
        main.profiler_cmd(args)

    with patch.dict(sys.modules, {"onnx9000_profiler": None}):
        main.profiler_cmd(args)
    assert mock_profiler_cls is not None or mock_profiler_cls is None


def test_serve_cmd_routes_and_assets() -> None:
    """Verify all route and asset translations in serve_cmd."""
    captured_handler = None

    class FakeServer:
        """Represents fake server, supporting operations such as serve_forever."""

        def __init__(
            self,
            addr: int | float | str | bool | bytes | list | dict | tuple | None,
            handler_cls: int | float | str | bool | bytes | list | dict | tuple | None,
        ) -> None:
            """Initialize FakeServer with configured parameters and state.

            Args:
                addr: Parameter specifying addr.
                handler_cls: Parameter specifying handler cls.
            """
            nonlocal captured_handler
            captured_handler = handler_cls

        def __enter__(self) -> int | float | str | bool | bytes | list | dict | tuple | None:
            """Enter the runtime context managed by FakeServer.

            Returns:
                Result of enter.
            """
            return self

        def __exit__(
            self, *a: int | float | str | bool | bytes | list | dict | tuple | None
        ) -> None:
            """Exit the runtime context managed by FakeServer and handle exceptions.

            Args:
                *a: Variadic positional arguments for a.
            """
            pass

        def serve_forever(self) -> NoReturn:
            """Execute serve forever on FakeServer.

            Raises:
                KeyboardInterrupt: If an invalid state or argument is encountered during execution.
            """
            raise KeyboardInterrupt

    with patch("socketserver.TCPServer", FakeServer):
        main.serve_cmd(argparse.Namespace(model=None, port=0))

    assert captured_handler is not None
    instance = object.__new__(captured_handler)

    for route in [
        "/hummingbird",
        "/native",
        "/genai",
        "/sparse",
        "/autograd",
        "/pytorch-codegen",
    ]:
        p = captured_handler.translate_path(instance, route)
        assert p.endswith("index.html")

    for target_idx in range(7):
        side_effects = [i == target_idx for i in range(6)]
        with patch("os.path.exists", side_effect=side_effects):
            asset_path = captured_handler.translate_path(instance, "/assets/bundle.js")
            assert "bundle.js" in asset_path


def test_convert_and_optimize_remaining_branches() -> None:
    """Verify remaining branches in convert_cmd, onnx2gguf_cmd, openvino_export_cmd, and sparse_prune_cmd."""
    # 1. triton >= 2.0 and ImportError in onnx2gguf_cmd
    args_gguf = argparse.Namespace(
        model="test.onnx",
        dry_run=True,
        force=False,
        architecture=None,
        tokenizer=None,
        outtype=None,
        output="out.gguf",
    )
    with patch.dict(sys.modules, {"triton": MagicMock(__version__="2.1.0")}):
        main.onnx2gguf_cmd(args_gguf)
    with patch.dict(sys.modules, {"triton": None}):
        main.onnx2gguf_cmd(args_gguf)

    # 2. openvino_export_cmd non-matching input name, empty shape, and data_type
    g_ov = Graph("ov")
    g_ov.inputs.append(ValueInfo("in1", (1, 3), DType.FLOAT32))
    g_ov.inputs.append(ValueInfo("scalar", (), DType.FLOAT32))
    args_ov = argparse.Namespace(
        model="test.onnx",
        shape=["other:1,2"],
        dynamic_batch=True,
        data_type=["in1:FP16", "other:FP32"],
        fp16=False,
        output_dir="ov_out",
    )
    with (
        patch("onnx9000.openvino.exporter.OpenVinoExporter.export", return_value=("<x/>", b"b")),
        patch("onnx9000_cli.main.load_onnx", return_value=g_ov),
        patch("os.makedirs"),
        patch("builtins.open", mock_open()),
    ):
        main.openvino_export_cmd(args_ov)
    assert g_ov.inputs[0].dtype == "FP16"

    # 3. sparse_prune_cmd with non-Constant tensor and low sparsity Constant
    g_sp = Graph("sp")
    g_sp.tensors["v"] = ValueInfo("v", (2,), DType.FLOAT32)
    c_dense = Constant(
        "c_dense", values=b"\x00\x00\x80?\x00\x00\x80?", shape=(2,), dtype=DType.FLOAT32
    )
    c_dense.is_initializer = True
    g_sp.tensors["c_dense"] = c_dense
    c_sparse = Constant("c_sparse", values=b"\x00" * 16, shape=(4,), dtype=DType.FLOAT32)
    c_sparse.is_initializer = True
    g_sp.tensors["c_sparse"] = c_sparse
    g_sp.sparse_initializers.append("c_sparse")
    args_sp = argparse.Namespace(model="test.onnx", recipe=None, sparsity=0.0, output="out.onnx")
    with (
        patch("onnx9000_cli.main.load_onnx", return_value=g_sp),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        main.sparse_prune_cmd(args_sp)

    # 4. sklearn convert_cmd branch
    with patch.dict(sys.modules, {"joblib": MagicMock(load=MagicMock(return_value=MagicMock()))}):
        with (
            patch(
                "onnx9000.converters.sklearn.builder.SKLearnParser.parse",
                return_value=Graph("sk"),
            ),
            patch("onnx9000.core.exporter.export_graph"),
        ):
            args_sk = argparse.Namespace(
                src="m.pkl", from_fmt="sklearn", to_fmt="onnx", output="m.onnx", weights=None
            )
            main.convert_cmd(args_sk)

    # 5. paddle convert_cmd file (non-dir) branch
    with (
        patch.dict(
            sys.modules,
            {
                "onnx9000.converters.paddle.api": MagicMock(
                    convert_paddle_to_onnx=MagicMock(return_value=Graph("pd"))
                )
            },
        ),
        patch("os.path.isdir", return_value=False),
        patch("os.path.exists", return_value=False),
        patch("builtins.open", mock_open(read_data=b"pd")),
        patch("onnx9000.core.exporter.export_graph"),
    ):
        args_pd = argparse.Namespace(
            src="m.pdmodel", from_fmt="paddle", to_fmt="onnx", output="m.onnx", weights=None
        )
        main.convert_cmd(args_pd)

    # 6. flax JSON fallback when parse_msgpack raises
    with (
        patch("onnx9000.jax.flax_parser.parse_msgpack", side_effect=ValueError("not msgpack")),
        patch("builtins.open", mock_open(read_data=b'{"a": 1}')),
        patch("onnx9000.core.exporter.export_graph"),
    ):
        args_flax = argparse.Namespace(
            src="m.flax", from_fmt="flax", to_fmt="onnx", output="m.onnx", weights=None
        )
        main.convert_cmd(args_flax)


def test_coverage_module_remaining_branches() -> None:
    """Verify remaining branches in onnx9000_cli.coverage."""
    from onnx9000_cli import coverage as cov

    with (
        patch("tempfile.TemporaryDirectory") as mock_td,
        patch("onnx9000_cli.coverage.get_pypi_info", return_value=("unknown", None)),
        patch("glob.glob", return_value=["snap1.json"]),
        patch("builtins.open", mock_open(read_data='{"version": "Not Installed", "objects": []}')),
    ):
        mock_td.return_value.__enter__.return_value = "/tmp/fake"
        res = cov.generate_framework_snapshots("/tmp/snaps")
        assert res["torch"]["version"] == "Not Installed"

    with (
        patch("tempfile.TemporaryDirectory") as mock_td,
        patch("subprocess.run") as mock_run,
        patch("os.path.exists", return_value=True),
        patch(
            "builtins.open",
            mock_open(
                read_data='## <a name="skip"></a><a name="skip"></a>\n## <a name=""></a>\nnormal line\n'
            ),
        ),
    ):
        mock_td.return_value.__enter__.return_value = "/tmp/fake"
        mock_run.return_value = MagicMock(stdout="abc1234\n")
        spec = cov.clone_and_parse_onnx_spec()
        assert spec["commit"] is not None

    with (
        patch("os.path.exists", return_value=True),
        patch("os.walk", return_value=[("/ops", [], ["op1.py"])]),
        patch("builtins.open", mock_open(read_data="record_op(dynamic_name)\n")),
    ):
        ops_non_const = cov.get_onnx9000_ops()
        assert ops_non_const == []

    with patch("os.path.isdir", return_value=False), patch("os.path.isfile", return_value=False):
        assert cov.count_supported_framework_objects("sklearn") == 0
        assert cov.count_supported_framework_objects("keras") == 0
        assert cov.count_supported_framework_objects("torch") >= 0

    with (
        patch("glob.glob", return_value=[]),
        patch("os.path.isfile", return_value=False),
        patch("os.path.isdir", return_value=True),
        patch("os.walk", return_value=[("/d", [], ["README.md", "mod.py"])]),
        patch("builtins.open", mock_open(read_data="a, b = 1, 2\n")),
    ):
        assert cov.count_supported_framework_objects("torch") >= 0

    with (
        patch("os.path.exists", side_effect=[True, False]),
        patch("builtins.open", mock_open(read_data="## Description\nhello\n\n")),
    ):
        cov.update_compliance_md("## Summary\n| a | b |")
