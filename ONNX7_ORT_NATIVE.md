# ONNX7: ONNX Runtime (ORT) Native Deployments

## Introduction
**Target Project:** Microsoft's [ONNX Runtime (Native Desktop/Server)](https://github.com/microsoft/onnxruntime)
**New Home:** `src/onnx9000/backends/cpu/`, `src/onnx9000/backends/cuda/`, `src/onnx9000/backends/apple/`, `src/onnx9000/backends/rocm/`

The standard ONNX Runtime is a monolithic C++ engine often exceeding 150MB because it statically links every ONNX operator, fallback kernel, and execution provider (EP) interface regardless of whether the specific model uses them. For local deployments (Python servers, desktop apps), this is massive overkill when you only need to run a ResNet or a Llama model on an NVIDIA GPU or Apple Silicon.

**The `onnx9000` Vision:** We are replacing the monolithic C++ runtime with a **dynamic, Zero-Overhead Python Dispatcher**. Because `onnx9000` already parses the `.onnx` graph into pure Python (`core/ir.py`), we don't need a heavy C++ runtime to orchestrate the graph. 

Instead, our Python execution engine dynamically evaluates the graph. When it hits a `MatMul` or `Conv`, it uses lightweight `ctypes`/`cffi` wrappers to call directly into the highly optimized hardware libraries (e.g., `cuBLAS`, `cuDNN`, Apple `Accelerate`, or `MPS`) *already present* on the host system. For custom operators, we dynamically JIT-compile a tiny, single-file C++ or CUDA kernel via PyTorch-like `load_inline` specifically for that model. The result is a native deployment that requires zero heavy pip dependencies (no `onnxruntime-gpu` wheel), runs just as fast, and deploys in milliseconds.

## Exhaustive Implementation Checklist (300+ Items)

### Phase 1: Pure Python Zero-Overhead Executor (Base CPU)
- [x] **Step 001:** Implement `onnx9000.backends.cpu.Executor` class.
- [x] **Step 002:** Implement a static memory planner in Python that allocates a single contiguous NumPy array (arena) for all activations.
- [x] **Step 003:** Implement offset mapping to slice the arena for individual tensors.
- [x] **Step 004:** Implement the execution loop iterating through topologically sorted `ir.Node` instances.
- [x] **Step 005:** Implement Python/NumPy bindings for `Add`, `Sub`, `Mul`, `Div`, `Pow`.
- [x] **Step 006:** Implement Python/NumPy bindings for `MatMul` (using `np.matmul`).
- [x] **Step 007:** Implement Python/NumPy bindings for `Conv` (using `np.correlate` or `im2col` + `matmul`).
- [x] **Step 008:** Implement Python/NumPy bindings for `Relu`, `Sigmoid`, `Tanh`, `Gelu`.
- [x] **Step 009:** Implement Python/NumPy bindings for `ReduceSum`, `ReduceMean`, `ReduceMax`.
- [x] **Step 010:** Implement Python/NumPy bindings for `Transpose`, `Reshape`, `Flatten`, `Concat`.
- [x] **Step 011:** Implement Python/NumPy bindings for `Gather`, `ScatterND`, `Slice`.
- [x] **Step 012:** Implement Python/NumPy bindings for `Softmax`, `LayerNorm`, `BatchNorm`.
- [x] **Step 013:** Implement thread-pool orchestration via Python `concurrent.futures` for independent DAG branches.
- [x] **Step 014:** Optimize the execution loop to avoid Python function call overhead where possible.
- [x] **Step 015:** Ensure exact numeric parity with standard `onnxruntime` CPU provider.
- [x] **Step 016:** Implement tensor caching for constants and initializers inside the executor.
- [x] **Step 017:** Handle dynamic shapes in the Python executor by dynamically slicing the memory arena.
- [x] **Step 018:** Implement fallback mechanisms if the arena size is insufficient (reallocation).
- [x] **Step 019:** Profile the Python execution loop on a standard ResNet-18 model.
- [x] **Step 020:** Optimize `im2col` implementation in NumPy to match basic C++ speeds.
- [x] **Step 021:** Finalize Phase 1 CPU Executor.

### Phase 2: NVIDIA CUDA & cuBLAS Dispatcher
- [x] **Step 022:** Create the `onnx9000.backends.cuda.Dispatcher`.
- [x] **Step 023:** Implement `ctypes` bindings for the `libcuda.so` driver API.
- [x] **Step 024:** Implement `cuMemAlloc`, `cuMemFree`, `cuMemcpyHtoD`, `cuMemcpyDtoH`.
- [x] **Step 025:** Implement asynchronous CUDA stream management (`cuStreamCreate`).
- [x] **Step 026:** Implement a CUDA memory pool in Python (managing `DevicePtr` allocations).
- [x] **Step 027:** Implement `ctypes` bindings for `libcublas.so`.
- [x] **Step 028:** Implement `cublasCreate`, `cublasDestroy`, `cublasSetStream`.
- [x] **Step 029:** Wrap `cublasSgemm` (FP32) for standard `MatMul` operations.
- [x] **Step 030:** Wrap `cublasHgemm` (FP16) for mixed-precision `MatMul`.
- [x] **Step 031:** Wrap `cublasSgemv` for Matrix-Vector multiplications.
- [x] **Step 032:** Implement `ctypes` bindings for `libcudnn.so`.
- [x] **Step 033:** Implement `cudnnCreate`, `cudnnSetTensorNdDescriptor`, `cudnnSetFilterNdDescriptor`.
- [x] **Step 034:** Wrap `cudnnConvolutionForward` for `Conv` operations.
- [x] **Step 035:** Wrap `cudnnPoolingForward` for `MaxPool` and `AveragePool`.
- [x] **Step 036:** Wrap `cudnnActivationForward` for `Relu`, `Sigmoid`, `Tanh`.
- [x] **Step 037:** Wrap `cudnnSoftmaxForward`.
- [x] **Step 038:** Wrap `cudnnBatchNormalizationForwardInference`.
- [x] **Step 039:** Implement a dynamic kernel JIT compiler (using `nvcc` via `subprocess`).
- [x] **Step 040:** Generate a standalone `.cu` file for fused elementwise operations (e.g., `Add` + `Relu`).
- [x] **Step 041:** Generate `.cu` kernels for operations lacking cuDNN support (e.g., specific `Gather`/`Scatter` layouts).
- [x] **Step 042:** Compile `.cu` to `.ptx` or `.cubin` natively at runtime.
- [x] **Step 043:** Load compiled `.ptx` kernels using `cuModuleLoad` and execute via `cuLaunchKernel`.
- [x] **Step 044:** Implement grid and block size calculation logic in Python based on tensor dimensions.
- [x] **Step 045:** Write tests verifying `cublasSgemm` output parity against PyTorch.
- [x] **Step 046:** Write tests verifying `cudnnConvolutionForward` output parity.
- [x] **Step 047:** Handle FP16 scaling and mixed-precision tensors seamlessly across the CUDA wrappers.
- [x] **Step 048:** Finalize Phase 2 CUDA Dispatcher.

### Phase 3: Apple Metal (MPS) & Accelerate Dispatcher
- [x] **Step 049:** Create the `onnx9000.backends.apple.Dispatcher`.
- [x] **Step 050:** Implement `ctypes` bindings for macOS `Accelerate.framework` (vDSP, vImage, BLAS).
- [x] **Step 051:** Wrap `cblas_sgemm` for fast CPU `MatMul` on Macs.
- [x] **Step 052:** Wrap `vDSP_vadd`, `vDSP_vsub`, `vDSP_vmul` for fast vectorized elementwise ops.
- [x] **Step 053:** Wrap `vDSP_vsmul` for scalar-vector multiplication.
- [x] **Step 054:** Implement `ctypes` bindings via PyObjC to interact with `Metal.framework`.
- [x] **Step 055:** Initialize the `MTLCreateSystemDefaultDevice`.
- [x] **Step 056:** Create an `MTLCommandQueue` and manage `MTLCommandBuffer` lifecycles.
- [x] **Step 057:** Implement a Metal buffer pool (`device.newBufferWithLength_options_`).
- [x] **Step 058:** Implement `ctypes` bindings to `MetalPerformanceShaders.framework` (MPS).
- [x] **Step 059:** Wrap `MPSMatrixMultiplication` for `MatMul`.
- [x] **Step 060:** Wrap `MPSCNNConvolution` for `Conv` operations.
- [x] **Step 061:** Wrap `MPSCNNPoolingMax`, `MPSCNNPoolingAverage`.
- [x] **Step 062:** Wrap `MPSCNNNeuronReLU`, `MPSCNNNeuronSigmoid`.
- [x] **Step 063:** Wrap `MPSCNNSoftMax`.
- [x] **Step 064:** Wrap `MPSCNNNormalization` (BatchNorm/LayerNorm equivalent).
- [x] **Step 065:** Dynamically generate MSL (Metal Shading Language) `.metal` strings for unsupported ops.
- [x] **Step 066:** Compile MSL strings at runtime using `device.newLibraryWithSource_options_error_`.
- [x] **Step 067:** Extract `MTLFunction` and create `MTLComputePipelineState`.
- [x] **Step 068:** Dispatch custom MSL kernels via `MTLComputeCommandEncoder`.
- [x] **Step 069:** Calculate optimal threadgroup counts (`dispatchThreadgroups_threadsPerThreadgroup_`).
- [x] **Step 070:** Handle memory synchronization (`MTLResourceStorageModeShared` vs `Managed`).
- [x] **Step 071:** Write tests verifying MPS output parity against standard PyTorch (MPS backend).
- [x] **Step 072:** Ensure seamless FP16 support natively on Apple Silicon via `float16_t` buffers.
- [x] **Step 073:** Finalize Phase 3 Apple Metal Dispatcher.

### Phase 4: AMD ROCm (HIP) & MIOpen Dispatcher
- [x] **Step 074:** Create the `onnx9000.backends.rocm.Dispatcher`.
- [x] **Step 075:** Implement `ctypes` bindings for the `libamdhip64.so` API.
- [x] **Step 076:** Implement `hipMalloc`, `hipFree`, `hipMemcpyHtoD`, `hipMemcpyDtoH`.
- [x] **Step 077:** Implement asynchronous HIP stream management (`hipStreamCreate`).
- [x] **Step 078:** Implement `ctypes` bindings for `librocblas.so`.
- [x] **Step 079:** Wrap `rocblas_sgemm` for standard FP32 `MatMul`.
- [x] **Step 080:** Wrap `rocblas_hgemm` for FP16 `MatMul`.
- [x] **Step 081:** Implement `ctypes` bindings for `libMIOpen.so`.
- [x] **Step 082:** Wrap `miopenConvolutionForward` for `Conv` operations.
- [x] **Step 083:** Wrap `miopenPoolingForward`.
- [x] **Step 084:** Wrap `miopenActivationForward`.
- [x] **Step 085:** Wrap `miopenSoftmaxForward`.
- [x] **Step 086:** Implement a dynamic kernel JIT compiler using `hipcc`.
- [x] **Step 087:** Generate standalone `.cpp` (HIP) files for fused elementwise operations.
- [x] **Step 088:** Compile `.cpp` to executable HSA code objects.
- [x] **Step 089:** Load compiled objects using `hipModuleLoad` and execute via `hipModuleLaunchKernel`.
- [x] **Step 090:** Write tests verifying `rocblas` output parity against PyTorch ROCm builds.
- [x] **Step 091:** Finalize Phase 4 ROCm Dispatcher.

### Phase 5: Native Execution Orchestrator & Benchmarks
- [x] **Step 092:** Implement `onnx9000.backends.runtime.NativeSession` class mirroring ORT `InferenceSession` API.
- [x] **Step 093:** Implement hardware auto-detection (NVIDIA SMI, `sysctl` for Apple Silicon, ROCm SMI).
- [x] **Step 094:** Automatically route the `ir.Graph` to the optimal `Dispatcher`.
- [x] **Step 095:** Implement device fallback logic (e.g., if a custom op lacks a CUDA kernel, copy to CPU, run, copy back).
- [x] **Step 096:** Implement a graph partitioning pass to separate CPU and GPU execution streams optimally.
- [x] **Step 097:** Implement explicit multi-GPU orchestration (splitting execution across multiple CUDA devices).
- [x] **Step 098:** Support user-provided configuration overrides (`device='cuda:0'`).
- [x] **Step 099:** Implement telemetry and detailed timeline profiling matching ORT capabilities.
- [x] **Step 100:** Profile end-to-end latency of `onnx9000` NativeSession vs `onnxruntime-gpu` for ResNet-50.
- [x] **Step 101:** Profile end-to-end latency of `onnx9000` NativeSession vs `onnxruntime-gpu` for BERT.
- [x] **Step 102:** Profile memory utilization (VRAM usage should be strictly defined by the tensor arena, completely removing ORT C++ engine overhead).
- [x] **Step 103:** Test dynamic JIT compilation overhead (compile once, cache globally).
- [x] **Step 104:** Ensure caching mechanism stores `.ptx` or `.metal` binaries persistently to disk.
- [x] **Step 105:** Write comprehensive documentation on 'Zero-Overhead Deployments' using `onnx9000`.
- [x] **Step 106:** Finalize Phase 5 Orchestrator and complete ONNX7 Native Deployments architecture.

