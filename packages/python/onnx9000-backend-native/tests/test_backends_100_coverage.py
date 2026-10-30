"""Exhaustive tests to reach 100% coverage across all 7 Python backend packages."""

from __future__ import annotations

import ctypes
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from onnx9000.backends.apple.executor import (
    Dispatcher as AppleDispatcher,
)
from onnx9000.backends.codegen.compiler import (
    compile_cpp,
    compile_static_lib,
    compile_wasm,
    load_ctypes_library,
    load_pybind_module,
)
from onnx9000.backends.codegen.generator import Generator
from onnx9000.backends.codegen.ops.tensor_ops import (
    generate_constant,
    generate_constant_of_shape,
)
from onnx9000.backends.codegen.utils import get_attribute, get_omp_pragma
from onnx9000.backends.cuda.compiler import CUDACompiler
from onnx9000.backends.cuda.cuda_arena import CUDAMemoryPlanner
from onnx9000.backends.cuda.executor import (
    Dispatcher as CudaDispatcher,
)
from onnx9000.backends.cuda.executor import (
    _cuda_add,
    _cuda_matmul,
    _cuda_mul,
    _cuda_sub,
)
from onnx9000.backends.ffi.core import (
    DynamicLibrary,
    DynamicLibraryError,
    HardwareContextHandle,
    get_cache_sizes,
    get_cpu_features,
)
from onnx9000.backends.memory.cpu_arena import CPUMemoryPlanner
from onnx9000.backends.memory.dlpack import from_dlpack
from onnx9000.backends.memory.metal_arena import MetalMemoryPlanner
from onnx9000.backends.rocm import bindings as rocm_bindings
from onnx9000.backends.rocm.compiler import ROCmCompiler
from onnx9000.backends.rocm.executor import Dispatcher as RocmDispatcher
from onnx9000.backends.rocm.executor import _rocm_matmul
from onnx9000.backends.testing.downloader import download_and_extract_onnx_tests
from onnx9000.backends.testing.runner import ONNXBackendTestRunner
from onnx9000.backends.webgpu.executor import WebGPUExecutionProvider
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import BackendExecutionError, CompilationError
from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor, ValueInfo
from onnx9000.openvino.api import export_model
from onnx9000.openvino.exporter import OpenVinoExporter
from onnx9000.tensorrt.builder import Builder, BuilderConfig, NetworkDefinition
from onnx9000.tensorrt.enums import DataType, MemoryPoolType
from onnx9000.tensorrt.ffi import TensorRTFFI, ffi
from onnx9000.tensorrt.network import IBuilderConfig, INetworkDefinition, ITensor
from onnx9000.tensorrt.ops_conv import trt_conv
from onnx9000.tensorrt.ops_dim import (
    _get_input,
    trt_concat,
    trt_gather,
    trt_reshape,
    trt_slice,
    trt_transpose,
)
from onnx9000.tensorrt.ops_matmul import trt_matmul
from onnx9000.tensorrt.registry import _TRT_OP_REGISTRY, get_op_translator, register_op
from onnx9000.tensorrt.structs import Dims


def test_apple_executor_branches() -> None:
    """Cover remaining branches in onnx9000-backend-apple/executor.py."""
    g = Graph("apple_g")
    g.inputs = ["missing_inp"]
    g.initializers = ["missing_init", "init_no_data", "init_bytes"]
    g.add_tensor(Tensor("init_no_data", (2,), DType.FLOAT32))
    g.add_tensor(Constant("init_bytes", values=b"\x00\x00\x80?", shape=(1,), dtype=DType.FLOAT32))
    g.add_node(Node("Relu", ["init_bytes"], ["missing_out_tensor"], {}, "r1"))

    disp = AppleDispatcher(g)
    with patch.object(
        disp.cpu_fallback,
        "execute",
        return_value={"np_out": np.ones((1,), dtype=np.float32)},
    ):
        disp._cpu_fallback_node(
            Node("Relu", ["init_bytes"], ["np_out", "unproduced_out"], {}, "r2")
        )

    with patch(
        "onnx9000.core.registry.global_registry.get_op",
        side_effect=ValueError("apple err"),
    ):
        disp._execute_node(Node("Relu", ["init_bytes"], ["missing_out_tensor"], {}, "r1"))


def test_cuda_compiler_arena_executor_branches(tmp_path: Path) -> None:
    """Cover remaining branches in onnx9000-backend-cuda."""
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        assert CUDACompiler.compile_kernel("code", "k1", cache_dir=str(tmp_path)) == b""
        assert CUDACompiler.compile_kernel("code", "k1_default_cache") == b""

    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["nvcc"], stderr=b"nvcc err"),
    ):
        with pytest.raises(BackendExecutionError, match="CUDA Kernel compilation failed"):
            CUDACompiler.compile_kernel("code", "k2", cache_dir=str(tmp_path))

    def write_ptx(cmd: list[str], **kwargs: object) -> None:
        """Test helper function write_ptx."""
        Path(cmd[-1]).write_bytes(b"ptx_data")

    with patch("subprocess.run", side_effect=write_ptx):
        assert CUDACompiler.compile_kernel("code", "k3", cache_dir=str(tmp_path)) == b"ptx_data"
        # Cached hit
        assert CUDACompiler.compile_kernel("code", "k3", cache_dir=str(tmp_path)) == b"ptx_data"

    # CUDAMemoryPlanner branches
    mock_cuda = MagicMock()
    mock_cuda.cuMemAlloc.return_value = 0
    mock_cuda.cuMemAlloc_v2.return_value = 0
    mock_cuda.cuMemHostAlloc.return_value = 0
    mock_cuda.cuMemAllocManaged.return_value = 0
    mock_cuda.cuMemcpyHtoD_v2.return_value = 0
    mock_cuda.cuMemcpyHtoDAsync_v2.return_value = 0
    mock_cuda.cuMemcpyDtoH_v2.return_value = 0
    mock_cuda.cuMemcpyDtoHAsync_v2.return_value = 0
    mock_cuda.cuStreamSynchronize.return_value = 0
    mock_cuda.cuMemFree.return_value = 0
    mock_cuda.cuMemFree_v2.return_value = 0

    with (
        patch("onnx9000.backends.cuda.cuda_arena.is_cuda_available", return_value=True),
        patch("onnx9000.backends.cuda.cuda_arena._cuda_lib", mock_cuda),
    ):
        planner = CUDAMemoryPlanner()
        planner.allocate_static("s1", 8, (2,), "float32")
        planner.build_arena()
        planner.arena_ptr = ctypes.c_void_p(100)
        assert planner.get_tensor_ptr("s1").value == 100
        with pytest.raises(BackendExecutionError, match="not found"):
            planner.get_tensor_ptr("missing")

        planner.allocate_pinned(16)
        planner.allocate_managed(16)
        planner.set_tensor("s1", np.zeros((2,), dtype=np.float32))
        planner.set_tensor("s1", np.zeros((2,), dtype=np.float32), stream=ctypes.c_void_p(1))
        _ = planner.get_host_tensor("s1")
        _ = planner.get_host_tensor("s1", stream=ctypes.c_void_p(1))
        with pytest.raises(BackendExecutionError, match="not found"):
            planner.get_host_tensor("missing")

        # Oversized static tensor (> 256 aligned bytes) triggers dynamic allocation
        planner.set_tensor("s1", np.zeros((100,), dtype=np.float32))
        planner.set_tensor("new_dyn", np.zeros((2,), dtype=np.float32))
        # Update existing dynamic allocation (same size and larger size)
        planner.set_tensor("new_dyn", np.zeros((2,), dtype=np.float32))
        planner.set_tensor("new_dyn", np.zeros((4,), dtype=np.float32))
        assert planner.get_tensor_ptr("new_dyn") is not None
        _ = planner.get_host_tensor("new_dyn")
        planner.allocate_dynamic("alloc_dyn", 8, (2,), "float32")
        planner.allocate_dynamic("alloc_dyn", 16, (4,), "float32")
        planner.synchronize_stream(ctypes.c_void_p(1))
        planner.cleanup()

        mock_cuda.cuMemFree_v2.side_effect = OSError("free error")
        planner.arena_ptr = ctypes.c_void_p(100)
        planner.__del__()

    with patch("onnx9000.backends.cuda.cuda_arena.is_cuda_available", return_value=False):
        planner_no_cuda = CUDAMemoryPlanner()
        planner_no_cuda.allocate_static("s_offline", 8, (2,), "float32")
        planner_no_cuda.arena_ptr = ctypes.c_void_p(100)
        planner_no_cuda.allocate_pinned(16)
        planner_no_cuda.allocate_managed(16)
        _ = planner_no_cuda.get_host_tensor("s_offline")
        planner_no_cuda.synchronize_stream(ctypes.c_void_p(1))
        planner_no_cuda.__del__()

    # CudaDispatcher branches
    from onnx9000.core.exceptions import UnsupportedOpError

    g = Graph("cuda_g")
    g.initializers = ["missing_init"]
    g.add_tensor(Tensor("dyn_t", ("batch", 4), DType.FLOAT32))
    g.add_tensor(Tensor("a", (2, 2), DType.FLOAT32))
    g.add_tensor(Tensor("b", (2, 2), DType.FLOAT32))
    g.add_tensor(Tensor("c", (2, 2), DType.FLOAT32))
    g.add_node(Node("Relu", ["a"], ["dyn_t", "missing_out"], {}, "r1"))

    with patch("onnx9000.backends.cuda.executor.is_cuda_available", return_value=False):
        disp = CudaDispatcher(g)
        disp._set_tensor_safe("a", np.ones((2, 2), dtype=np.float32))
        disp._set_tensor_safe("b", np.ones((2, 2), dtype=np.float32))
        with patch.object(
            disp.cpu_fallback,
            "execute",
            return_value={
                "c": np.ones((2, 2), dtype=np.float32),
                "c_tensor": Tensor(
                    "c_tensor",
                    (2, 2),
                    DType.FLOAT32,
                    data=np.ones((2, 2), dtype=np.float32).tobytes(),
                ),
            },
        ):
            disp._cpu_fallback_node(
                Node("Add", ["a", "b"], ["c", "c_tensor", "unproduced"], {}, "add1")
            )

        with patch(
            "onnx9000.core.registry.global_registry.get_op",
            side_effect=UnsupportedOpError("Add"),
        ):
            disp.initialized = True
            disp._execute_node(Node("Add", ["a", "b"], ["c"], {}, "add1"))
            disp.initialized = False

        with patch(
            "onnx9000.core.registry.global_registry.get_op",
            side_effect=ValueError("cuda err"),
        ):
            disp.initialized = True
            disp._execute_node(Node("Add", ["a", "b"], ["c"], {}, "add1"))
            disp.initialized = False

        with patch(
            "onnx9000.core.registry.global_registry.get_op",
            side_effect=RuntimeError("cuda fatal"),
        ):
            disp.initialized = True
            with pytest.raises(RuntimeError, match="cuda fatal"):
                disp._execute_node(Node("Add", ["a", "b"], ["c"], {}, "add1"))
            disp.initialized = False

        with (
            patch.object(disp, "_execute_matmul") as m_mm,
            patch.object(disp, "_execute_elementwise") as m_ew,
        ):
            n = Node("Add", ["a", "b"], ["c"], {}, "n")
            _cuda_matmul(disp, n)
            _cuda_add(disp, n)
            _cuda_sub(disp, n)
            _cuda_mul(disp, n)
            assert m_mm.called and m_ew.call_count == 3

        # __del__ with exceptions
        disp.initialized = True
        disp.cublas_handle = ctypes.c_void_p(1)
        disp.cudnn_handle = ctypes.c_void_p(1)
        disp.stream = ctypes.c_void_p(1)
        mock_err = MagicMock()
        mock_err.cublasDestroy_v2.side_effect = OSError("err")
        mock_err.cudnnDestroy.side_effect = OSError("err")
        mock_err.cuStreamDestroy_v2.side_effect = OSError("err")
        with (
            patch("onnx9000.backends.cuda.executor.is_cublas_available", return_value=True),
            patch("onnx9000.backends.cuda.executor.is_cudnn_available", return_value=True),
            patch("onnx9000.backends.cuda.executor.is_cuda_available", return_value=True),
            patch("onnx9000.backends.cuda.executor._cublas_lib", mock_err),
            patch("onnx9000.backends.cuda.executor._cudnn_lib", mock_err),
            patch("onnx9000.backends.cuda.executor._cuda_lib", mock_err),
        ):
            disp.__del__()
        disp.initialized = False


def test_rocm_bindings_and_executor_branches(tmp_path: Path) -> None:
    """Cover all missed lines and branches in onnx9000-backend-rocm."""
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        assert ROCmCompiler.compile_kernel("code", "rk1", cache_dir=str(tmp_path)) == b""
        assert ROCmCompiler.compile_kernel("code", "rk1_default") == b""

    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["hipcc"], stderr=b"hipcc err"),
    ):
        with pytest.raises(BackendExecutionError, match="ROCm Kernel compilation failed"):
            ROCmCompiler.compile_kernel("code", "rk2", cache_dir=str(tmp_path))

    def write_hsaco(cmd: list[str], **kwargs: object) -> None:
        """Test helper function write_hsaco."""
        Path(cmd[-1]).write_bytes(b"hsaco_data")

    with patch("subprocess.run", side_effect=write_hsaco):
        assert ROCmCompiler.compile_kernel("code", "rk3", cache_dir=str(tmp_path)) == b"hsaco_data"
        assert ROCmCompiler.compile_kernel("code", "rk3", cache_dir=str(tmp_path)) == b"hsaco_data"

    mock_lib = MagicMock()
    rocm_bindings._register_hip_api(mock_lib)
    rocm_bindings._register_rocblas_api(mock_lib)
    rocm_bindings._register_miopen_api(mock_lib)

    rocm_bindings.check_hip_error(0)
    rocm_bindings.check_rocblas_error(0)
    rocm_bindings.check_miopen_error(0)
    with pytest.raises(BackendExecutionError):
        rocm_bindings.check_hip_error(1)
    with pytest.raises(BackendExecutionError):
        rocm_bindings.check_rocblas_error(1)
    with pytest.raises(BackendExecutionError):
        rocm_bindings.check_miopen_error(1)

    g = Graph("rocm_g")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    g.initializers = ["w", "missing_init"]
    g.add_tensor(
        Constant(
            "w",
            values=np.ones((2, 2), dtype=np.float32).tobytes(),
            shape=(2, 2),
            dtype=DType.FLOAT32,
        )
    )
    g.add_tensor(Tensor("dyn_t", ("batch", 2), DType.FLOAT32))
    g.add_tensor(Tensor("c", (2, 2), DType.FLOAT32))
    g.add_node(Node("MatMul", ["a", "b"], ["c", "dyn_t", "missing_t"], {}, "mm"))

    # Dispatcher when is_hip_available() is False
    with patch("onnx9000.backends.rocm.executor.is_hip_available", return_value=False):
        disp_offline = RocmDispatcher(g)
        arr = np.ones((2, 2), dtype=np.float32)
        disp_offline.run({"a": arr, "b": arr})
        disp_offline.__del__()

    mock_hip = MagicMock()
    mock_hip.hipStreamCreate.return_value = 0
    mock_hip.hipStreamDestroy.return_value = 0
    mock_rocblas = MagicMock()
    mock_rocblas.rocblas_create_handle.return_value = 0
    mock_rocblas.rocblas_set_stream.return_value = 0
    mock_rocblas.rocblas_destroy_handle.return_value = 0
    mock_miopen = MagicMock()
    mock_miopen.miopenCreate.return_value = 0
    mock_miopen.miopenDestroy.return_value = 0

    with (
        patch("onnx9000.backends.rocm.executor.is_hip_available", return_value=True),
        patch("onnx9000.backends.rocm.executor.is_rocblas_available", return_value=True),
        patch("onnx9000.backends.rocm.executor.is_miopen_available", return_value=True),
        patch("onnx9000.backends.rocm.executor._hip_lib", mock_hip),
        patch("onnx9000.backends.rocm.executor._rocblas_lib", mock_rocblas),
        patch("onnx9000.backends.rocm.executor._miopen_lib", mock_miopen),
    ):
        disp = RocmDispatcher(g)
        disp.stream = ctypes.c_void_p(1)
        disp.rocblas_handle = ctypes.c_void_p(1)
        disp.miopen_handle = ctypes.c_void_p(1)

        arr = np.ones((2, 2), dtype=np.float32)
        res = disp.run({"a": arr, "b": arr})
        assert "c" in res
        # Run a second time so 'a' and 'b' are already in dynamic_allocations
        disp.run({"a": arr, "b": arr})

        # _cpu_fallback_node with Tensor having .data and with np.ndarray
        with patch.object(
            disp.cpu_fallback,
            "execute",
            return_value={
                "new_out": Tensor("new_out", (2, 2), DType.FLOAT32, data=arr.tobytes()),
                "np_out": arr,
            },
        ):
            disp._cpu_fallback_node(
                Node("Relu", ["a"], ["new_out", "np_out", "unproduced"], {}, "r")
            )

        # _execute_node with RuntimeError and ValueError
        with patch(
            "onnx9000.core.registry.global_registry.get_op",
            side_effect=RuntimeError("fatal"),
        ):
            with pytest.raises(RuntimeError, match="fatal"):
                disp._execute_node(Node("MatMul", ["a", "b"], ["c"], {}, "mm"))

        with patch(
            "onnx9000.core.registry.global_registry.get_op",
            side_effect=ValueError("unsupported"),
        ):
            disp._execute_node(Node("MatMul", ["a", "b"], ["c"], {}, "mm"))

        # __del__ normal and with exceptions
        disp.__del__()
        mock_hip.hipStreamDestroy.side_effect = OSError("err")
        mock_rocblas.rocblas_destroy_handle.side_effect = OSError("err")
        mock_miopen.miopenDestroy.side_effect = OSError("err")
        disp.__del__()

        # Also test when hip is available but rocblas/miopen are False
        with (
            patch("onnx9000.backends.rocm.executor.is_rocblas_available", return_value=False),
            patch("onnx9000.backends.rocm.executor.is_miopen_available", return_value=False),
        ):
            disp2 = RocmDispatcher(g)
            with pytest.raises(BackendExecutionError, match="rocBLAS is required"):
                _rocm_matmul(disp2, Node("MatMul", ["a", "b"], ["c"], {}, "mm"))
            disp2.__del__()


def test_webgpu_executor_branches() -> None:
    """Cover remaining branch in onnx9000-backend-webgpu/executor.py."""
    ep = WebGPUExecutionProvider()
    ep.initialize()
    g = Graph("wgpu_g")
    assert ep.execute(g, MagicMock(), {}) == {}


def test_openvino_exporter_and_api_branches() -> None:
    """Cover remaining lines and branches in onnx9000-openvino."""
    g = Graph("ov_g")
    g.inputs = [
        ValueInfo("unused_in", (1, 2), DType.FLOAT32),
        ValueInfo("x", (1, 2, 4, 4), DType.FLOAT32),
        ValueInfo("scalar_in", (), DType.FLOAT32),
    ]
    g.outputs = [
        ValueInfo("unmapped_out", (1,), DType.FLOAT32),
        ValueInfo("final_out", (1,), DType.FLOAT32),
    ]
    g.initializers = ["missing_init", "w_conv", "w_conv_dup", "b_conv", "w_empty", "w_scalar"]
    w_bytes = struct.pack("<2f", 1.0, 2.0)
    g.add_tensor(Constant("w_conv", values=w_bytes, shape=(1, 1, 1, 2), dtype=DType.FLOAT32))
    g.add_tensor(Constant("w_conv_dup", values=w_bytes, shape=(1, 1, 1, 2), dtype=DType.FLOAT32))
    g.add_tensor(
        Constant("b_conv", values=struct.pack("<1f", 0.5), shape=(1,), dtype=DType.FLOAT32)
    )
    g.add_tensor(Constant("w_empty", values=None, shape=(1,), dtype=DType.FLOAT32))
    g.add_tensor(
        Constant("w_scalar", values=struct.pack("<1f", 1.0), shape=(), dtype=DType.FLOAT32)
    )

    # Add nodes covering all remaining op mappings in OpenVinoExporter
    g.add_node(
        Node(
            "Conv",
            ["x", "w_conv", "b_conv"],
            ["conv1"],
            {
                "group": Attribute("group", "INT", 2),
                "strides": Attribute("strides", "INTS", [1, 1]),
                "dilations": Attribute("dilations", "INTS", [1, 1]),
                "pads": Attribute("pads", "INTS", [0, 0, 0, 0, 0, 0]),
                "auto_pad": Attribute("auto_pad", "STRING", "SAME_UPPER"),
            },
            "conv_grouped",
        )
    )
    g.add_node(
        Node(
            "ConvTranspose",
            ["conv1", "w_conv_dup"],
            ["conv_tr1"],
            {
                "group": Attribute("group", "INT", 2),
                "output_padding": Attribute("output_padding", "INTS", [0, 0]),
            },
            "conv_tr_grouped",
        )
    )
    g.add_node(Node("ConvTranspose", ["conv_tr1", "w_conv"], ["conv_tr2"], {}, "conv_tr_plain"))
    g.add_node(
        Node(
            "AveragePool",
            ["conv_tr2"],
            ["pool1"],
            {
                "kernel_shape": Attribute("kernel_shape", "INTS", [2, 2]),
                "strides": Attribute("strides", "INTS", [1, 1]),
                "pads": Attribute("pads", "INTS", [0, 0, 0, 0, 0, 0]),
                "auto_pad": Attribute("auto_pad", "STRING", "VALID"),
                "count_include_pad": Attribute("count_include_pad", "INT", 1),
            },
            "avg_pool",
        )
    )
    g.add_node(
        Node(
            "Gelu",
            ["pool1"],
            ["gelu1"],
            {"approximate": Attribute("approximate", "STRING", "none")},
            "gelu_erf",
        )
    )
    g.add_node(Node("Gelu", ["gelu1"], ["gelu2"], {}, "gelu_default"))
    g.add_node(
        Node(
            "Pad",
            ["gelu2"],
            ["pad1"],
            {
                "mode": Attribute("mode", "STRING", "constant"),
                "pads": Attribute("pads", "INTS", [0, 0, 1, 1]),
            },
            "pad_opset10",
        )
    )
    g.add_node(Node("Pad", ["pad1", "w_conv", "w_scalar"], ["pad2"], {}, "pad_3inp"))
    g.add_node(
        Node(
            "Gather",
            ["pad2", "w_scalar"],
            ["gath1"],
            {
                "batch_dims": Attribute("batch_dims", "INT", 1),
                "axis": Attribute("axis", "INT", 0),
            },
            "gather_node",
        )
    )
    g.add_node(Node("Slice", ["gath1"], ["slice1"], {}, "slice_node"))
    g.add_node(
        Node(
            "ReduceMean",
            ["slice1"],
            ["red1"],
            {
                "keepdims": Attribute("keepdims", "INT", 1),
                "axes": Attribute("axes", "INTS", [0]),
            },
            "reduce_node",
        )
    )
    g.add_node(
        Node(
            "ArgMax",
            ["red1"],
            ["arg1"],
            {
                "keepdims": Attribute("keepdims", "INT", 0),
                "axis": Attribute("axis", "INT", 0),
            },
            "argmax_node",
        )
    )
    g.add_node(
        Node(
            "Resize",
            ["arg1"],
            ["res1"],
            {
                "mode": Attribute("mode", "STRING", "nearest"),
                "coordinate_transformation_mode": Attribute(
                    "coordinate_transformation_mode", "STRING", "half_pixel"
                ),
                "nearest_mode": Attribute("nearest_mode", "STRING", "floor"),
            },
            "resize_node",
        )
    )
    g.add_node(
        Node(
            "DepthToSpace",
            ["res1"],
            ["dts1"],
            {
                "blocksize": Attribute("blocksize", "INT", 2),
                "mode": Attribute("mode", "STRING", "DCR"),
            },
            "dts_node",
        )
    )
    g.add_node(
        Node(
            "NonMaxSuppression",
            ["dts1"],
            ["nms1"],
            {"center_point_box": Attribute("center_point_box", "INT", 1)},
            "nms_center",
        )
    )
    g.add_node(Node("NonMaxSuppression", ["nms1"], ["nms2"], {}, "nms_default"))
    g.add_node(
        Node(
            "RoiAlign",
            ["nms2"],
            ["roi1"],
            {"mode": Attribute("mode", "STRING", "max")},
            "roi_max",
        )
    )
    g.add_node(Node("RoiAlign", ["roi1"], ["roi2"], {}, "roi_default"))
    g.add_node(Node("QuantizeLinear", ["roi2"], ["ql1"], {}, "ql_node"))
    g.add_node(
        Node(
            "Einsum",
            ["ql1"],
            ["ein1"],
            {"equation": Attribute("equation", "STRING", "ij->ji")},
            "einsum_node",
        )
    )
    g.add_node(
        Node(
            "LayerNormalization",
            ["ein1"],
            ["ln1"],
            {
                "axis": Attribute("axis", "INT", -1),
                "epsilon": Attribute("epsilon", "FLOAT", 1e-5),
            },
            "ln_node",
        )
    )
    g.add_node(
        Node(
            "InstanceNormalization",
            ["ln1"],
            ["in1"],
            {"epsilon": Attribute("epsilon", "FLOAT", 1e-5)},
            "in_node",
        )
    )
    g.add_node(
        Node(
            "LpNormalization",
            ["in1"],
            ["lp1"],
            {"axis": Attribute("axis", "INT", 1), "p": Attribute("p", "INT", 2)},
            "lp_node",
        )
    )
    g.add_node(
        Node(
            "BatchNormalization",
            ["lp1"],
            ["bn1"],
            {"epsilon": Attribute("epsilon", "FLOAT", 1e-5)},
            "bn_node",
        )
    )
    g.add_node(Node("Dropout", ["bn1"], ["drop1"], {}, "drop_node"))
    sub_g = Graph("sub")
    g.add_node(
        Node(
            "If",
            ["drop1"],
            ["if1"],
            {
                "then_branch": Attribute("then_branch", "GRAPH", sub_g),
                "else_branch": Attribute("else_branch", "GRAPH", sub_g),
            },
            "if_node",
        )
    )
    g.add_node(
        Node(
            "Loop",
            ["if1"],
            ["loop1"],
            {"body": Attribute("body", "GRAPH", sub_g)},
            "loop_node",
        )
    )
    g.add_node(
        Node(
            "Cast",
            ["loop1"],
            ["cast1"],
            {"to": Attribute("to", "INT", DType.FLOAT32.value)},
            "cast_node",
        )
    )
    g.add_node(
        Node(
            "GridSample",
            ["cast1"],
            ["gs1"],
            {
                "mode": Attribute("mode", "STRING", "bilinear"),
                "padding_mode": Attribute("padding_mode", "STRING", "zeros"),
                "align_corners": Attribute("align_corners", "INT", 1),
            },
            "gs_node",
        )
    )
    g.add_node(Node("Size", ["gs1"], ["size1"], {}, "size_node"))
    g.add_node(Node("Flatten", ["size1"], ["flat1"], {}, "flat_node"))
    g.add_node(
        Node(
            "Transpose",
            ["flat1"],
            ["tr1"],
            {"perm": Attribute("perm", "INTS", [1, 0])},
            "tr_node",
        )
    )
    g.add_node(
        Node(
            "GatherElements",
            ["tr1"],
            ["ge1"],
            {"axis": Attribute("axis", "INT", 0)},
            "ge_node",
        )
    )
    g.add_node(
        Node(
            "ConstantOfShape",
            ["ge1"],
            ["cos_f32"],
            {
                "value": Attribute(
                    "value",
                    "TENSOR",
                    Constant("v1", values=struct.pack("<f", 1.0), shape=(1,), dtype=DType.FLOAT32),
                )
            },
            "cos1",
        )
    )
    g.add_node(
        Node(
            "ConstantOfShape",
            ["cos_f32"],
            ["cos_i64"],
            {
                "value": Attribute(
                    "value",
                    "TENSOR",
                    Constant("v2", values=struct.pack("<q", 2), shape=(1,), dtype=DType.INT64),
                )
            },
            "cos2",
        )
    )
    g.add_node(
        Node(
            "ConstantOfShape",
            ["cos_i64"],
            ["cos_i32"],
            {
                "value": Attribute(
                    "value",
                    "TENSOR",
                    Constant("v3", values=struct.pack("<i", 3), shape=(1,), dtype=DType.INT32),
                )
            },
            "cos3",
        )
    )
    g.add_node(
        Node(
            "Add",
            ["cos_i32", "cos_i32"],
            ["final_out"],
            {},
            "dup_edge_add",
        )
    )
    g.add_node(Node("Identity", ["scalar_in", ""], [], {}, "no_out_node"))

    # Add nodes without optional attributes to cover all false branches in OpenVinoExporter
    for idx, op_name in enumerate(
        [
            "Softmax",
            "Gather",
            "ReduceMean",
            "ArgMax",
            "Resize",
            "DepthToSpace",
            "Einsum",
            "LayerNormalization",
            "InstanceNormalization",
            "LpNormalization",
            "BatchNormalization",
            "If",
            "Loop",
            "Cast",
            "GridSample",
            "GatherElements",
        ]
    ):
        g.add_node(Node(op_name, ["x"], [f"no_attr_{idx}"], {}, f"no_attr_{op_name}"))
    # Gather with axis and 3 inputs (len(inputs_to_map) != 2)
    g.add_node(
        Node(
            "Gather",
            ["x", "w_scalar", "w_scalar"],
            ["gath_3in"],
            {"axis": Attribute("axis", "INT", 0)},
            "gath_3in",
        )
    )
    # Size with empty inputs
    g.add_node(Node("Size", [], ["size_empty_in"], {}, "size_empty"))
    # ConstantOfShape with DType.BOOL value tensor
    g.add_node(
        Node(
            "ConstantOfShape",
            ["x"],
            ["cos_bool"],
            {
                "value": Attribute(
                    "value",
                    "TENSOR",
                    Constant("v_bool", values=b"\x01", shape=(1,), dtype=DType.BOOL),
                )
            },
            "cos_bool",
        )
    )
    # Conv with empty bias input name (not in port_ids)
    g.add_node(Node("Conv", ["x", "w_conv", ""], ["conv_empty_bias"], {}, "conv_empty_bias"))

    xml_str, bin_bytes = export_model(g, precision="fp16", clamp_dynamic=True)
    assert "<net" in xml_str and len(bin_bytes) > 0

    exp = OpenVinoExporter(g)
    exp._emit_dynamic_const("c_i32", [1], [], DType.INT32)
    exp._emit_dynamic_const("c_i32_dup", [1], [], DType.INT32)
    with pytest.raises(CompilationError, match="not implemented"):
        exp._emit_dynamic_const("c_bool", [1], [1], DType.BOOL)
    with pytest.raises(CompilationError, match="Unsupported dtype"):
        exp._map_dtype(DType.STRING)

    g_bad = Graph("bad")
    g_bad.add_node(Node("Relu", ["missing_ptr"], ["y"], {}, "r"))
    with pytest.raises(CompilationError, match="Missing input pointer"):
        OpenVinoExporter(g_bad).export()


def test_tensorrt_branches() -> None:
    """Cover all remaining lines and branches in onnx9000-tensorrt."""
    # FFI loading on win32 vs linux, plugin lib, version extraction, logging callback, check_error
    mock_cdll = MagicMock()
    mock_cdll.getInferLibVersion.return_value = 10201
    with (
        patch.object(sys, "platform", "win32"),
        patch("ctypes.CDLL", return_value=mock_cdll),
    ):
        trt_ffi = TensorRTFFI()
        assert trt_ffi.version == (10, 2, 1)
        for sev in [0, 1, 2, 3, 4]:
            trt_ffi._c_log_callback(None, sev, b"test message")
        trt_ffi.check_error(0, "ok")
        with pytest.raises(BackendExecutionError, match="TensorRT Error"):
            trt_ffi.check_error(1, "bad")

    # Test _load_library when plugin lib lacks initLibNvInferPlugins and when self.lib is None
    mock_no_init = MagicMock(spec=[])
    with (
        patch.object(sys, "platform", "win32"),
        patch("ctypes.CDLL", side_effect=[mock_no_init, mock_no_init]),
    ):
        TensorRTFFI()
    with (
        patch.object(sys, "platform", "win32"),
        patch("ctypes.CDLL", side_effect=[OSError("no nvinfer"), mock_no_init]),
    ):
        TensorRTFFI()

    # BuilderConfig, NetworkDefinition, Builder
    mock_lib = MagicMock()
    with patch.object(ffi, "lib", mock_lib):
        ffi._setup_logging_callback()
        # BuilderConfig with setMemoryPoolLimit, setMaxWorkspaceSize only, and with neither
        cfg = BuilderConfig(123)
        cfg.set_memory_pool_limit(MemoryPoolType.kWORKSPACE, 1024)
        del mock_lib.setMemoryPoolLimit
        cfg.set_memory_pool_limit(MemoryPoolType.kWORKSPACE, 1024)
        del mock_lib.setMaxWorkspaceSize
        cfg.set_memory_pool_limit(MemoryPoolType.kWORKSPACE, 1024)

        net_def = NetworkDefinition(456)
        net_def.destroy()
        del mock_lib.destroyNetworkDefinition
        net_def2 = NetworkDefinition(456)
        net_def2.destroy()

        # Builder error and success branches
        with patch.object(ffi, "lib", None):
            with pytest.raises(BackendExecutionError, match="not loaded"):
                Builder()

        del mock_lib.createInferBuilder_INTERNAL
        with pytest.raises(BackendExecutionError, match="createInferBuilder_INTERNAL not found"):
            Builder()

        mock_lib.createInferBuilder_INTERNAL = MagicMock(return_value=0)
        with pytest.raises(BackendExecutionError, match="Failed to create TensorRT Builder"):
            Builder()

        mock_lib.createInferBuilder_INTERNAL = MagicMock(return_value=999)
        with patch.object(ffi, "version", (10, 0, 0)):
            b = Builder()
            mock_lib.createNetworkV2 = MagicMock(return_value=777)
            assert b.create_network().ptr == 777
            del mock_lib.createNetworkV2
            with pytest.raises(BackendExecutionError, match="createNetworkV2 not found"):
                b.create_network()
            mock_lib.createNetworkV2 = MagicMock(return_value=0)
            with pytest.raises(BackendExecutionError, match="Failed to create NetworkDefinition"):
                b.create_network()

            mock_lib.createBuilderConfig = MagicMock(return_value=888)
            assert b.create_builder_config().ptr == 888
            del mock_lib.createBuilderConfig
            with pytest.raises(BackendExecutionError, match="createBuilderConfig not found"):
                b.create_builder_config()

            b.destroy()
            b.destroy()  # second call hits not self.ptr
            del mock_lib.destroyInferBuilder
            b2 = Builder()
            b2.destroy()

        # INetworkDefinition & IBuilderConfig & ITensor
        t = ITensor(11, "t1")
        assert repr(t) == "<ITensor name='t1'>"

        inet = INetworkDefinition(222)
        mock_lib.addInput = MagicMock(return_value=55)
        assert inet.add_input("x", DataType.kFLOAT, Dims([1, 2])).ptr == 55
        del mock_lib.addInput
        with pytest.raises(BackendExecutionError, match="addInput not found"):
            inet.add_input("x", DataType.kFLOAT, Dims([1, 2]))
        mock_lib.addInput = MagicMock(return_value=0)
        with pytest.raises(BackendExecutionError, match="Failed to add input"):
            inet.add_input("x", DataType.kFLOAT, Dims([1, 2]))

        mock_lib.markOutput = MagicMock()
        inet.mark_output(t)
        del mock_lib.markOutput
        with pytest.raises(BackendExecutionError, match="markOutput not found"):
            inet.mark_output(t)

        mock_lib.destroyNetworkDefinition = MagicMock()
        inet.destroy()
        inet.destroy()
        del mock_lib.destroyNetworkDefinition
        inet2 = INetworkDefinition(223)
        inet2.destroy()

        mock_lib.destroyBuilderConfig = MagicMock()
        icfg = IBuilderConfig(333)
        icfg.destroy()
        icfg.destroy()
        del mock_lib.destroyBuilderConfig
        icfg2 = IBuilderConfig(334)
        icfg2.destroy()

        # ops_conv, ops_matmul, ops_dim, registry, structs
        with pytest.raises(BackendExecutionError, match="cannot exceed 8"):
            Dims([1] * 9)

        @register_op("", "CustomDuplicateOp")
        def _op1() -> None:
            """Test helper function _op1."""
            pass

        @register_op("", "CustomDuplicateOp")
        def _op2() -> None:
            """Test helper function _op2."""
            pass

        assert get_op_translator("", "CustomDuplicateOp") is _op2
        _TRT_OP_REGISTRY.pop(("", "CustomDuplicateOp"), None)
        with pytest.raises(BackendExecutionError, match="No TensorRT translation"):
            get_op_translator("", "NonExistentTrtOp")

        mock_lib.addConvolutionNd = MagicMock(return_value=10)
        trt_conv(inet, Node("Conv", ["t1", "w"], ["y"]), {"t1": t})
        mock_lib.addConvolutionNd = MagicMock(return_value=0)
        with pytest.raises(BackendExecutionError, match="Failed to add Conv"):
            trt_conv(inet, Node("Conv", ["t1", "w"], ["y"]), {"t1": t})
        del mock_lib.addConvolutionNd
        with pytest.raises(BackendExecutionError, match="addConvolutionNd not found"):
            trt_conv(inet, Node("Conv", ["t1", "w"], ["y"]), {"t1": t})

        mock_lib.addMatrixMultiply = MagicMock(return_value=20)
        trt_matmul(inet, Node("MatMul", ["t1", "t1"], ["y"]), {"t1": t})
        mock_lib.addMatrixMultiply = MagicMock(return_value=0)
        with pytest.raises(BackendExecutionError, match="Failed to add MatMul"):
            trt_matmul(inet, Node("MatMul", ["t1", "t1"], ["y"]), {"t1": t})
        del mock_lib.addMatrixMultiply
        with pytest.raises(BackendExecutionError, match="addMatrixMultiply not found"):
            trt_matmul(inet, Node("MatMul", ["t1", "t1"], ["y"]), {"t1": t})

        assert _get_input(Node("Reshape", [], ["y"]), {}, 0) is None
        mock_lib.addShuffle = MagicMock(return_value=1)
        trt_reshape(inet, Node("Reshape", ["t1"], ["y"]), {"t1": t})
        trt_transpose(inet, Node("Transpose", ["t1"], ["y"], {"perm": [1, 0]}), {"t1": t})
        with pytest.raises(BackendExecutionError, match="Missing input for Reshape"):
            trt_reshape(inet, Node("Reshape", [], ["y"]), {})
        del mock_lib.addShuffle
        with pytest.raises(BackendExecutionError, match="addShuffle not found"):
            trt_reshape(inet, Node("Reshape", ["t1"], ["y"]), {"t1": t})
        with pytest.raises(BackendExecutionError, match="addShuffle not found"):
            trt_transpose(inet, Node("Transpose", ["t1"], ["y"]), {"t1": t})

        mock_lib.addConcatenation = MagicMock(return_value=1)
        trt_concat(inet, Node("Concat", ["t1"], ["y"], {"axis": 0}), {"t1": t})
        del mock_lib.addConcatenation
        with pytest.raises(BackendExecutionError, match="addConcatenation not found"):
            trt_concat(inet, Node("Concat", ["t1"], ["y"]), {"t1": t})

        mock_lib.addSlice = MagicMock(return_value=1)
        trt_slice(inet, Node("Slice", ["t1"], ["y"]), {"t1": t})
        del mock_lib.addSlice
        with pytest.raises(BackendExecutionError, match="addSlice not found"):
            trt_slice(inet, Node("Slice", ["t1"], ["y"]), {"t1": t})

        mock_lib.addGather = MagicMock(return_value=1)
        trt_gather(inet, Node("Gather", ["t1", "t1"], ["y"], {"axis": 1}), {"t1": t})
        del mock_lib.addGather
        with pytest.raises(BackendExecutionError, match="addGather not found"):
            trt_gather(inet, Node("Gather", ["t1", "t1"], ["y"]), {"t1": t})


def test_backend_native_remaining_branches(tmp_path: Path) -> None:
    """Cover remaining lines and branches in onnx9000-backend-native."""
    from onnx9000.backends.codegen.utils import sanitize_name
    from onnx9000.core import config

    # utils.py
    n_attr = Node("Relu", ["x"], ["y"], {"alpha": Attribute("alpha", "FLOAT", 2.0)}, "r")
    assert get_attribute(n_attr, "alpha") == 2.0
    assert sanitize_name("123abc") == "var_123abc"
    with patch.object(config, "ONNX9000_ENABLE_LOOP_UNROLLING", True, create=True):
        assert "unroll" in get_omp_pragma("8")
    with patch.object(config, "ONNX9000_USE_CUDA", True, create=True):
        assert get_omp_pragma("8") == ""

    # tensor_ops.py: generate_constant without value, with data=None, or with list data; generate_constant_of_shape without value or with non-Constant value
    g = Graph("t_ops")
    g.add_tensor(Tensor("c_out", (2,), DType.FLOAT32))
    ctx = Generator(g)
    assert "Constant" in generate_constant(Node("Constant", [], ["c_out"], {}), ctx)
    assert "Constant" in generate_constant(
        Node(
            "Constant",
            [],
            ["c_out"],
            {
                "value": Attribute(
                    "value", "TENSOR", Tensor("null_t", (2,), DType.FLOAT32, data=None)
                )
            },
        ),
        ctx,
    )
    list_tensor = Tensor("lt", (2,), DType.FLOAT32)
    list_tensor.data = [1.0, 2.0]
    assert "Constant" in generate_constant(
        Node("Constant", [], ["c_out"], {"value": Attribute("value", "TENSOR", list_tensor)}),
        ctx,
    )
    empty_list_tensor = Tensor("elt", (0,), DType.FLOAT32)
    empty_list_tensor.data = []
    assert "Constant" in generate_constant(
        Node("Constant", [], ["c_out"], {"value": Attribute("value", "TENSOR", empty_list_tensor)}),
        ctx,
    )
    g.add_tensor(Tensor("in_shape", (1,), DType.INT64))
    assert "ConstantOfShape" in generate_constant_of_shape(
        Node("ConstantOfShape", ["in_shape"], ["c_out"], {}),
        ctx,
    )
    assert "ConstantOfShape" in generate_constant_of_shape(
        Node(
            "ConstantOfShape",
            ["in_shape"],
            ["c_out"],
            {
                "value": Attribute(
                    "value",
                    "TENSOR",
                    Constant("c_null", values=None, shape=(1,), dtype=DType.FLOAT32),
                )
            },
        ),
        ctx,
    )
    assert "ConstantOfShape" in generate_constant_of_shape(
        Node(
            "ConstantOfShape",
            ["in_shape"],
            ["c_out"],
            {
                "value": Attribute(
                    "value",
                    "TENSOR",
                    Constant("c_1byte", values=b"\x01", shape=(1,), dtype=DType.UINT8),
                )
            },
        ),
        ctx,
    )

    # generator.py: mmap_constants, embed_constants with values/None/data, write_to_directory(run_clang_format=False)
    g_gen = Graph("gen_g")
    g_gen.inputs = ["x", "x"]
    g_gen.outputs = ["y", "y2"]
    g_gen.initializers = ["w_f32", "w_i64", "w_i32", "w_bool", "w_none", "w_bytes"]
    g_gen.add_tensor(Tensor("x", (2,), DType.FLOAT32))
    g_gen.add_tensor(Tensor("y", (2,), DType.FLOAT32))
    g_gen.add_tensor(Tensor("y2", (2,), DType.FLOAT32))
    g_gen.add_tensor(Tensor("dead_out", (2,), DType.FLOAT32))

    def make_val_tensor(name: str, dtype: DType, vals: list | None) -> Tensor:
        """Test helper function make_val_tensor."""
        t = Tensor(name, (2,), dtype, is_initializer=True, data=None)
        t.values = vals
        return t

    g_gen.add_tensor(make_val_tensor("w_f32", DType.FLOAT32, [1.0, 2.0]))
    g_gen.add_tensor(make_val_tensor("w_i64", DType.INT64, [1, 2]))
    g_gen.add_tensor(make_val_tensor("w_i32", DType.INT32, [1, 2]))
    g_gen.add_tensor(make_val_tensor("w_bool", DType.BOOL, [True, False]))
    g_gen.add_tensor(make_val_tensor("w_none", DType.FLOAT32, None))
    g_gen.add_tensor(
        Constant("w_bytes", values=b"\x00\x00\x80?\x00\x00\x00@", shape=(2,), dtype=DType.FLOAT32)
    )
    g_gen.add_node(Node("Relu", ["x"], ["y"], {}, "r1"))
    g_gen.add_node(Node("Relu", ["x"], ["y2"], {}, "r2"))
    g_gen.add_node(Node("Relu", ["x"], ["dead_out"], {}, "dead_node"))

    gen = Generator(g_gen)
    mmap_bin = tmp_path / "weights.bin"
    code_mmap = gen.generate(mmap_constants=True, mmap_file=str(mmap_bin))
    assert "GeneratedModel" in code_mmap and mmap_bin.exists()
    code_embed = gen.generate(embed_constants=True)
    assert "GeneratedModel" in code_embed

    with patch("subprocess.run", side_effect=FileNotFoundError()):
        cpp_file = gen.write_to_directory(str(tmp_path / "out_dir"), run_clang_format=True)
        assert Path(str(cpp_file)).exists()
    cpp_file2 = gen.write_to_directory(str(tmp_path / "out_dir2"), run_clang_format=False)
    assert Path(str(cpp_file2)).exists()

    # compiler.py branches
    with patch("subprocess.run") as mock_run:
        with patch.object(sys, "platform", "linux"):
            compile_cpp("int main() {}", use_pybind=False, extra_flags=["-O2"])
        with patch.dict(sys.modules, {"pybind11": None}):
            assert compile_cpp("int main() {}", use_pybind=True) is None

        wasm_js = tmp_path / "out.js"
        wasm_bin = tmp_path / "out.wasm"
        wasm_bin.write_bytes(b"0" * (5 * 1024 * 1024))  # > 4MB triggers warning
        with pytest.warns(UserWarning, match="exceeds standard 4MB"):
            compile_wasm(
                "int main() {}",
                output_path=str(wasm_js),
                extra_flags=["-O2"],
                standalone_wasm=False,
                enable_simd=True,
                use_pthreads=True,
                initial_memory=65536,
                maximum_memory=131072,
                emit_tsd=True,
            )
        # compile_wasm when wasm_path does not exist
        compile_wasm("int main() {}", output_path=str(tmp_path / "nonexistent.js"))

        with patch("importlib.util.spec_from_file_location", return_value=None):
            with pytest.raises(ImportError):
                load_pybind_module("non_existent.so")

        with patch("ctypes.cdll.LoadLibrary", return_value="mock_lib"):
            assert load_ctypes_library("lib.so") == "mock_lib"

        # compile_static_lib success and failure branches
        def fake_static_run(cmd: list[str], **kwargs: object) -> None:
            """Test helper function fake_static_run."""
            for i, arg in enumerate(cmd):
                if arg == "-o" and i + 1 < len(cmd):
                    Path(cmd[i + 1]).write_bytes(b"obj")

        mock_run.side_effect = fake_static_run
        with patch.object(sys, "platform", "win32"):
            compile_static_lib("int x = 0;")
        with patch.object(sys, "platform", "darwin"):
            compile_static_lib(
                "int x = 0;", output_path=str(tmp_path / "lib.a"), extra_flags=["-O2"]
            )

        mock_run.side_effect = subprocess.CalledProcessError(1, ["clang++"], stderr="obj err")
        with pytest.raises(BackendExecutionError, match="Compilation failed"):
            compile_cpp("int main() {}", use_pybind=False)
        with pytest.raises(BackendExecutionError, match="Object compilation failed"):
            compile_static_lib("int x = 0;")

        def fail_on_ar(cmd: list[str], **kwargs: object) -> None:
            """Test helper function fail_on_ar."""
            if cmd[0] in ("ar", "lib"):
                raise subprocess.CalledProcessError(1, cmd, stderr="ar err")

        mock_run.side_effect = fail_on_ar
        with pytest.raises(BackendExecutionError, match="Archiving failed"):
            compile_static_lib("int x = 0;")

    # ffi/core.py branches
    with (
        patch.dict("os.environ", {"ONNX9000_LIB_CUSTOM": "/fake/libcustom.so"}),
        patch("ctypes.CDLL", return_value=MagicMock()),
    ):
        lib_custom = DynamicLibrary("custom")
        fn = lib_custom.define("my_fn", [], ctypes.c_int)
        assert lib_custom.my_fn is fn
        assert callable(lib_custom.uncached_fn)

    with patch("ctypes.CDLL") as m_cdll:
        m_cdll.side_effect = [MagicMock(spec=[])]
        lib_win = DynamicLibrary("c")
        lib_win.os = "Windows"
        with pytest.raises(AttributeError):
            lib_win.define("missing_sym", [], ctypes.c_int)

    with (
        patch.object(ctypes, "RTLD_GLOBAL", None),
        patch("onnx9000.backends.ffi.core.hasattr", return_value=False),
        patch("ctypes.CDLL", return_value=MagicMock()),
    ):
        lib = DynamicLibrary("c")
        with patch("ctypes.CDLL") as m_cdll:
            m_cdll.return_value.dlerror.return_value = b""
            del lib.lib.missing_sym
            with pytest.raises(AttributeError):
                lib.define("missing_sym", [], ctypes.c_int)

    with (
        patch.object(ctypes, "RTLD_GLOBAL", None),
        patch("onnx9000.backends.ffi.core.hasattr", return_value=False),
        patch("ctypes.CDLL") as m_cdll2,
    ):
        m_cdll2.return_value = None
        with pytest.raises(DynamicLibraryError):
            DynamicLibrary("none_lib", versions=[1, 2])

    HardwareContextHandle._cleanup(None, MagicMock())
    with (
        patch("platform.system", return_value="Darwin"),
        patch(
            "subprocess.check_output",
            return_value=b"""hw.l1dcachesize: 32768
hw.l2cachesize: 262144
hw.l3cachesize: 8388608
other: 1
""",
        ),
    ):
        sizes = get_cache_sizes()
        assert sizes["l3"] == 8388608

    with (
        patch("platform.system", return_value="Linux"),
        patch("os.path.exists", side_effect=lambda p: "index1" in p or "index2" in p),
        patch(
            "builtins.open",
            MagicMock(
                side_effect=[
                    MagicMock(__enter__=lambda s: MagicMock(read=lambda: "32K")),
                    MagicMock(__enter__=lambda s: MagicMock(read=lambda: "65536")),
                ]
            ),
        ),
    ):
        linux_sizes = get_cache_sizes()
        assert linux_sizes["l1"] == 32768 and linux_sizes["l2"] == 65536

    with patch("platform.system", return_value="Windows"):
        assert get_cpu_features() == {"avx": False, "avx2": False, "avx512": False, "neon": False}
        assert get_cache_sizes() == {"l1": 0, "l2": 0, "l3": 0}

    # memory/cpu_arena.py branches
    cpu_p = CPUMemoryPlanner()
    with pytest.raises(BackendExecutionError, match="not found in CPU memory planner"):
        cpu_p.get_tensor_ptr("missing")
    with patch("platform.system", return_value="Windows"):
        mm = cpu_p._allocate_mmap(0)
        mm.close()
    with (
        patch("platform.system", return_value="Linux"),
        patch("ctypes.CDLL") as m_libc,
    ):
        mm2 = cpu_p._allocate_mmap(2 * 1024 * 1024)
        mm2.close()
        m_libc.side_effect = OSError("no libc")
        mm3 = cpu_p._allocate_mmap(2 * 1024 * 1024)
        mm3.close()
    with patch("mmap.mmap", side_effect=OSError("oom")):
        with pytest.raises(MemoryError):
            cpu_p._allocate_mmap(4096)

    cpu_p.allocate_static("s1", 8, (2,), "float32")
    cpu_p.build_arena()
    assert cpu_p.get_tensor_ptr("s1") is not None
    # set_tensor when size_bytes > static size, and when already in tensors_shape_dtype
    cpu_p.set_tensor("s1", memoryview(b"0" * 16), (4,), "float32")
    assert cpu_p.get_tensor_ptr("s1") is not None
    assert len(cpu_p.get_host_tensor("s1")) == 16
    cpu_p.add_ref("s1")
    cpu_p.add_ref("missing")
    cpu_p.release_ref("s1")
    del cpu_p.dynamic_sizes["s1"]
    cpu_p.release_ref("s1")  # ref_count reaches 0 when name not in dynamic_sizes
    cpu_p.release_ref("missing")
    cpu_p.allocate_dynamic("d1", 8, (2,), "float32")
    cpu_p.allocate_dynamic("d1", 8, (2,), "float32")  # reallocates existing dynamic tensor

    # cleanup BufferError handling
    mock_mm = MagicMock()
    mock_mm.close.side_effect = BufferError("exported")
    cpu_p.arena_mmap = mock_mm
    cpu_p.dynamic_allocations["d_err"] = mock_mm
    cpu_p.cleanup()

    # memory/dlpack.py: from_dlpack when strides is None (null pointer)
    arr = np.ones((2, 2), dtype=np.float32)

    class NullStrideDLPack:
        """Test helper class NullStrideDLPack."""

        def __dlpack__(self) -> object:
            """Test helper function __dlpack__."""
            cap = arr.__dlpack__()
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            ptr = ctypes.pythonapi.PyCapsule_GetPointer(cap, b"dltensor")
            from onnx9000.backends.memory.dlpack import DLManagedTensor

            managed = DLManagedTensor.from_address(ptr)
            managed.dl_tensor.strides = None
            return cap

    _, shape, strides, _, _ = from_dlpack(NullStrideDLPack())
    assert shape == (2, 2) and strides is None

    # memory/metal_arena.py
    mp = MetalMemoryPlanner()
    mp.build_arena()  # current_offset == 0 branch
    with pytest.raises(BackendExecutionError, match="not found in Metal memory planner"):
        mp.get_tensor_ptr("missing")
    with pytest.raises(BackendExecutionError, match="not found"):
        mp.get_host_tensor("missing")
    mp.allocate_static("ms1", 4, (1,), "float32")
    mp.build_arena()
    # Oversized static tensor in MetalMemoryPlanner
    mp.set_tensor("ms1", memoryview(b"12345678"), (2,), "float32")
    mp.set_tensor("ms1", memoryview(b"12345678"), (2,), "float32")
    assert mp.get_tensor_ptr("ms1") is not None
    mp.set_tensor("dyn_m", memoryview(b"1234"), (1,), "float32")
    mp.set_tensor("dyn_m", memoryview(b"1234"), (1,), "float32")
    assert len(mp.get_host_tensor("dyn_m")) == 4
    mp.cleanup()

    # testing/downloader.py & runner.py
    import zipfile

    dl_dir = tmp_path / "dl_zip"
    dl_dir.mkdir()
    zip_file = dl_dir / "onnx-v1.14.0.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        zf.writestr("onnx-1.14.0/onnx/backend/test/data/node/dummy.txt", "ok")
    assert download_and_extract_onnx_tests(dl_dir).exists()

    tests_dir = tmp_path / "onnx-1.14.0" / "onnx" / "backend" / "test" / "data" / "node"
    tests_dir.mkdir(parents=True)
    assert download_and_extract_onnx_tests(tmp_path) == tests_dir

    node_dir = tmp_path / "node_test_int"
    node_dir.mkdir()
    (node_dir / "model.onnx").write_bytes(b"dummy")
    ds_dir = node_dir / "test_data_set_0"
    ds_dir.mkdir()

    runner = ONNXBackendTestRunner(providers=[])
    int_tensor = Tensor("x", (1,), DType.INT32, data=struct.pack("<i", 7))
    bool_tensor = Tensor("b", (1,), DType.BOOL, data=b"\x01")
    mock_sess = MagicMock()
    mock_sess.get_inputs.return_value = [ValueInfo("x", (1,), DType.INT32)]
    mock_sess.get_outputs.return_value = [
        ValueInfo("y1", (1,), DType.INT32),
        ValueInfo("y2", (1,), DType.BOOL),
    ]
    mock_sess.run.return_value = [int_tensor, bool_tensor]
    with (
        patch("onnx9000.core.parser.core.load", return_value=Graph("g")),
        patch("onnx9000.backends.testing.runner.InferenceSession", return_value=mock_sess),
        patch.object(runner, "load_tensors", return_value=[int_tensor, bool_tensor]),
    ):
        ok, msg = runner.run_node_test(node_dir)
        assert ok and msg == "Passed"
