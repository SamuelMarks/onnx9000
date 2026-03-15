# ONNX4: Olive (Model Optimizer) Native Rewrite

## Introduction
**Target Project:** Microsoft's [Olive](https://github.com/microsoft/Olive)
**New Home:** `src/onnx9000/optimize/hardware/`

The official Olive framework is an advanced, hardware-aware model optimization toolkit. It orchestrates a complex pipeline of external tools (ONNX Runtime quantization, DirectML, TensorRT, Neural Magic SparseML) to compress and accelerate models. While powerful, Olive depends on a massive environment setup, Docker containers, and heavy C++ binaries that cannot be executed locally in a browser or lightweight client.

**The `onnx9000` Vision:** We are rebuilding the core optimization logic (Quantization, Weight Packing, Memory Layouts) into a pure Python toolchain. Our optimizer focuses specifically on **Web Delivery & Edge Execution**, compressing models heavily for HTTP streaming and restrictive WebWorker memory limits. By writing hardware-aware passes in pure Python, we can dynamically quantize models inside the browser or immediately upon export without relying on heavy external C++ optimizers.

## Exhaustive Implementation Checklist (300+ Items)

### Phase 1: Pure Python INT8 Quantization Algorithms
- [x] **Step 001:** Create the base `onnx9000.optimize.hardware.Quantizer` class.
- [x] **Step 002:** Implement Min-Max (Asymmetric) Quantization algorithm in pure Python.
- [x] **Step 003:** Implement Min-Max (Symmetric) Quantization algorithm in pure Python.
- [x] **Step 004:** Implement pure Python scale and zero-point calculation for tensors.
- [x] **Step 005:** Implement `DynamicQuantizeLinear` calculation natively.
- [x] **Step 006:** Implement `QuantizeLinear` calculation natively.
- [x] **Step 007:** Implement `DequantizeLinear` calculation natively.
- [x] **Step 008:** Write logic to insert `QuantizeLinear` -> `DequantizeLinear` pairs around operators (Fake Quantization).
- [x] **Step 009:** Implement `MatMulInteger` conversion natively.
- [x] **Step 010:** Implement `ConvInteger` conversion natively.
- [x] **Step 011:** Implement `QLinearConv` (fully quantized convolution) fusion.
- [x] **Step 012:** Implement `QLinearMatMul` (fully quantized matrix multiplication) fusion.
- [x] **Step 013:** Implement `QLinearAdd` fusion.
- [x] **Step 014:** Implement `QLinearSigmoid` fusion.
- [x] **Step 015:** Implement `QLinearLeakyRelu` fusion.
- [x] **Step 016:** Implement activation clipping (e.g., Relu6) prior to quantization.
- [x] **Step 017:** Implement per-tensor (scalar scale/zero-point) quantization.
- [x] **Step 018:** Implement per-channel (vector scale/zero-point) quantization.
- [x] **Step 019:** Handle axis alignment for per-channel quantization.
- [x] **Step 020:** Ensure `QLinearConv` zero-points match ONNX specifications (uint8 vs int8 constraints).
- [x] **Step 021:** Implement pure Python cross-entropy calibration for Post-Training Quantization (PTQ).
- [x] **Step 022:** Implement KL-Divergence (Entropy) calibration for PTQ.
- [x] **Step 023:** Implement Percentile (e.g., 99.9%) calibration for PTQ.
- [x] **Step 024:** Write a data loader interface to stream calibration data into the Python PTQ loop.
- [x] **Step 025:** Implement moving-average statistics gathering for PTQ.
- [x] **Step 026:** Implement layer-wise quantization error analysis (MSE between float and int8 outputs).
- [x] **Step 027:** Implement a 'skip layer' heuristic if quantization MSE exceeds a given threshold.
- [x] **Step 028:** Convert initializer weights to INT8 natively in Python (compressing the Protobuf).
- [x] **Step 029:** Write parity tests comparing Python quantized weights vs `onnxruntime.quantization`.
- [x] **Step 030:** Ensure all INT8 quantized graphs pass `onnx.checker`.
- [x] **Step 031:** Handle `Bias` tensors in quantized Convs (typically quantized to INT32).
- [x] **Step 032:** Implement INT32 accumulation scaling for `MatMulInteger` -> `Add` sequences.
- [x] **Step 033:** Implement ONNX `DynamicQuantizeMatMul` fusion.
- [x] **Step 034:** Finalize Phase 1 pure Python INT8 Quantization algorithms.
- [x] **Step 035:** Write tests verifying symmetric zero-points are exactly 0.
- [x] **Step 036:** Write tests verifying asymmetric zero-points fit in uint8.
- [x] **Step 037:** Support FP16 (Float16) quantization (casting weights without scaling).
- [x] **Step 038:** Implement BF16 (Bfloat16) conversion logic.
- [x] **Step 039:** Write an exporter converting FP32 `ModelProto` to FP16 `ModelProto` natively.
- [x] **Step 040:** Ensure graph I/O types remain unchanged during weight-only quantization.
- [x] **Step 041:** Implement a dynamic dispatcher choosing between symmetric/asymmetric based on weight distribution.
- [x] **Step 042:** Finalize Phase 1 Quantization Framework.

### Phase 2: INT4 Sub-Byte Quantization & Packing
- [x] **Step 043:** Implement standard INT4 Asymmetric Quantization in pure Python.
- [x] **Step 044:** Implement standard INT4 Symmetric Quantization in pure Python.
- [x] **Step 045:** Implement 4-bit weight packing natively in Python (2 weights per uint8 byte).
- [x] **Step 046:** Handle little-endian vs big-endian byte ordering during packing.
- [x] **Step 047:** Implement group-wise quantization (e.g., groups of 32, 64, or 128 weights sharing a scale).
- [x] **Step 048:** Implement `BlockQuantizeLinear` logic.
- [x] **Step 049:** Implement ONNX `MatMulNBits` (Opset 21/Microsoft extension) native generation.
- [x] **Step 050:** Write logic to generate INT4 unpacked fallback graphs (using INT8 nodes + shifts) for standard WebGPU.
- [x] **Step 051:** Implement AWQ (Activation-aware Weight Quantization) calibration in pure Python.
- [x] **Step 052:** Implement GPTQ (Accurate Post-Training Quantization) algorithm in pure Python.
- [x] **Step 053:** Write tests comparing AWQ/GPTQ outputs to reference implementations.
- [x] **Step 054:** Implement SmoothQuant algorithms (shifting difficulty from activations to weights).
- [x] **Step 055:** Write native JS/WASM decoding logic for packed INT4 weights during inference.
- [x] **Step 056:** Write native WGSL shaders for unpacking INT4 and performing `MatMul` directly.
- [x] **Step 057:** Optimize WGSL INT4 shaders using bitwise shifts and masks (`>>`, `&`).
- [x] **Step 058:** Handle edge cases where matrix dimensions are not multiples of the block size.
- [x] **Step 059:** Profile INT4 vs INT8 memory bandwidth utilization in WebGPU.
- [x] **Step 060:** Write a Python utility to analyze the sparsity of weight matrices.
- [x] **Step 061:** Implement Sparse INT8 packing (e.g., 2:4 sparsity pattern).
- [x] **Step 062:** Implement WGSL shaders supporting 2:4 sparse matrix multiplication.
- [x] **Step 063:** Test INT4 LLaMA-style model conversion running exclusively in Pyodide.
- [x] **Step 064:** Ensure INT4 export correctly tags custom opsets.
- [x] **Step 065:** Finalize Phase 2 INT4 Sub-Byte Quantization.

### Phase 3: Hardware-Aware Memory Layout Optimization
- [x] **Step 066:** Implement memory layout transformation passes in `onnx9000.optimize.hardware.layout`.
- [x] **Step 067:** Implement NCHW (Channels First) to NHWC (Channels Last) conversion pass.
- [x] **Step 068:** Implement NHWC to NCHW conversion pass.
- [x] **Step 069:** Detect operators supporting `layout` attributes natively (e.g., Conv).
- [x] **Step 070:** Inject `Transpose` nodes where layout mismatch occurs.
- [x] **Step 071:** Implement a greedy `Transpose` cancellation pass (fusing adjacent transposes).
- [x] **Step 072:** Implement a pass to push `Transpose` nodes down the graph through elementwise operations.
- [x] **Step 073:** Write heuristic matching WebGPU optimal layouts (NHWC usually preferred for cache locality).
- [x] **Step 074:** Write heuristic matching WASM SIMD optimal layouts (NCHW usually preferred).
- [x] **Step 075:** Implement memory alignment packing (e.g., padding channels to multiples of 4 for `vec4<f32>` in WGSL).
- [x] **Step 076:** Write a pass that pads all tensor shapes to alignment boundaries.
- [x] **Step 077:** Update `Conv`, `MatMul`, and `Reshape` parameters mathematically to account for alignment padding.
- [x] **Step 078:** Implement constant unfolding (converting highly dimensional constants to 1D flat arrays for WGSL).
- [x] **Step 079:** Implement a memory estimation pass (simulating VRAM usage before execution).
- [x] **Step 080:** Implement a pass detecting and resolving WebGPU `maxStorageBufferBindingSize` limits by chunking large tensors.
- [x] **Step 081:** Support generating specialized graphs for iOS CoreML / Neural Engine via layout hints.
- [x] **Step 082:** Support generating specialized graphs for Android NNAPI layout hints.
- [x] **Step 083:** Write parity tests ensuring NHWC transformed graphs produce identical outputs to NCHW graphs.
- [x] **Step 084:** Test Transpose cancellation pass thoroughly against complex branching graphs.
- [x] **Step 085:** Finalize Phase 3 Memory Layout Architecture.

### Phase 4: Execution Pipelining & WebWorker Optimizations
- [x] **Step 086:** Implement a pass to split massive graphs into smaller subgraphs for WebWorker execution.
- [x] **Step 087:** Identify independent execution paths (branches) in the DAG.
- [x] **Step 088:** Partition the graph into separate `ir.Graph` objects communicating via SharedArrayBuffer.
- [x] **Step 089:** Implement a scheduling algorithm (e.g., Critical Path Method) to orchestrate partitioned subgraphs.
- [x] **Step 090:** Implement memory pooling hints natively inside the ONNX graph.
- [x] **Step 091:** Inject custom `Alloc` and `Free` ONNX nodes to explicitly control WebGPU/WASM memory.
- [x] **Step 092:** Write a heuristic to determine if an operation should run on CPU (WASM) vs GPU (WebGPU) based on payload size.
- [x] **Step 093:** Implement automatic device-placement passes.
- [x] **Step 094:** Implement a pass to merge multiple tiny operations into a single massive `Einsum` or `FusedOp` for GPU dispatch.
- [x] **Step 095:** Implement a pass that pre-calculates static shapes for dynamic shape graphs when inputs are bounded.
- [x] **Step 096:** Generate a 'Static Graph' completely devoid of shape inference at runtime.
- [x] **Step 097:** Implement an auto-tuner in Python: running the graph via `onnxruntime` with different layouts/quantizations and recording latency.
- [x] **Step 098:** Implement a genetic algorithm auto-tuner to find optimal layer-by-layer quantization schemes.
- [x] **Step 099:** Integrate the auto-tuner natively into the JS WebWorker (for client-specific hardware tuning).
- [x] **Step 100:** Write tests validating the partitioned graphs produce identical outputs to the monolithic graph.
- [x] **Step 101:** Ensure custom `Alloc/Free` nodes pass ONNX schema validation (as a custom domain).
- [x] **Step 102:** Implement fallback mechanisms if the auto-tuner crashes.
- [x] **Step 103:** Finalize Phase 4 Graph Execution Pipeline.

### Phase 5: Seamless API Integration & Polish
- [x] **Step 104:** Create the high-level `onnx9000.optimize.optimize(graph, target='webgpu')` API.
- [x] **Step 105:** Create `onnx9000.optimize.quantize_dynamic(graph)` API matching `onnxruntime.quantization`.
- [x] **Step 106:** Create `onnx9000.optimize.quantize_static(graph, calibration_data)` API.
- [x] **Step 107:** Ensure the optimizer runs cleanly inside Pyodide.
- [x] **Step 108:** Implement a JS wrapper to run the Python optimizer directly in the browser.
- [x] **Step 109:** Write comprehensive documentation on 'How to compress models for the Web'.
- [x] **Step 110:** Build a CLI tool: `onnx9000-cli optimize input.onnx output_int8.onnx --target=webgpu`.
- [x] **Step 111:** Support parsing standard ONNX Runtime JSON configuration files (matching Olive configs).
- [x] **Step 112:** Generate an optimization report (Original Size, Final Size, Estimated VRAM, Optimization Passes applied).
- [x] **Step 113:** Implement visual DAG comparison (Before vs After optimization) in the UI.
- [x] **Step 114:** Write unit tests verifying model file size reduction is >50% for INT8 and >70% for INT4.
- [x] **Step 115:** Test end-to-end: PyTorch Model -> Trace -> Optimize -> Quantize -> WebGPU Execute.
- [x] **Step 116:** Publish the `@onnx9000/optimize` package.
- [x] **Step 117:** Finalize Phase 5 and the ONNX4 Hardware Optimizer Architecture.

