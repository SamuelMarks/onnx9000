# onnx9000 Roadmap

**🔥 STATUS:** `onnx9000` is maturing rapidly. Over 200 standard ONNX operators have been implemented. The core IR parsing, autograd engine, static C++ memory planning, Apple Accelerate framework, WebAssembly SIMD backends, CUDA target, WebGPU, and advanced WebWorker RPC architectures are fully integrated and verified with 100% test and doc coverage across Python, C++, and TypeScript.


| Phase | Feature / Goal | Description | Complexity | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Full Im2Col Convolution** | Upgrade the naive `Conv` C++ template to use a highly optimized `im2col` + `MatMul` approach, supporting arbitrary strides, padding, and dilations. | Medium | ✅ |
| **2** | **WASM SIMD Intrinsics** | Inject Emscripten `#pragma clang loop vectorize(enable)` and WebAssembly SIMD (`-msimd128`) flags into the Jinja2 templates for 3x-5x browser speedups. | Medium | ✅ |
| **3** | **Shape Broadcasting Engine** | Implement a robust C++ macro/template system for dynamic N-dimensional NumPy-style broadcasting during element-wise operations. | High | ✅ |
| **4** | **Standard Operator Parity** | Expand `onnx9000.codegen.ops` to cover the top 100 most common ONNX Opset 18/19 operations (Gather, Slice, Split, Pad, LSTM, etc.). | High | ✅ |
| **5** | **Reverse-Mode Optimizer Generators** | Add Jinja2 templates that natively generate C++ weight-update logic (SGD, AdamW) to be called after the `ReluGrad`/`MatMulGrad` backward pass execution. | Medium | ✅ |
| **6** | **Constant Folding & Graph Fusions** | Implement the deferred optimization passes in `parser/passes.py` to pre-compute constants and fuse `Conv + BatchNorm` or `MatMul + Add` before C++ generation. | Medium | ✅ |
| **7** | **Memory Liveness Re-use Algorithm** | Upgrade the naive memory planner to accurately map non-overlapping tensor lifespans into shared buffer IDs, drastically reducing WASM memory footprints. | High | ✅ |
| **8** | **Hardware: Apple Accelerate / Metal** | Add a compilation target that swaps the standard C++ `<cmath>` loops with Apple Accelerate `vDSP` calls or pure Metal shaders for native Mac performance. | Very High | ✅ |
| **9** | **Hardware: CUDA C++ Target** | Create a parallel Jinja2 environment (`codegen/cuda`) that translates the `ir.Graph` directly into `.cu` files, wrapping custom PTX kernels or cuBLAS calls. | Very High | ✅ |
| **10** | **Control Flow (If, Loop)** | Implement support for ONNX Subgraphs (`If`, `Loop`, `Scan`), requiring the C++ generator to emit dynamic branching and recursive scope evaluation. | Extreme | ✅ |


## The Grand Unification (V2 Roadmap)

With the core transpiler and execution engine verified, `onnx9000` is expanding to absorb the functionality of 18 major ONNX ecosystem projects natively into Python and JS. See [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) and individual project files for detailed 300+ item checklists:

* **[Phase 11: ONNX Runtime Training](ONNX0_ORT_TRAINING.md)**
* **[Phase 12: ONNX Runtime Web](ONNX1_ORT_WEB.md)**
* **[Phase 13: ONNX Runtime Extensions](ONNX2_ORT_EXTENSIONS.md)**
* **[Phase 14: Torch/TF Exporters](ONNX3_TORCH_EXPORTERS.md)**
* **[Phase 15: Olive Optimizer](ONNX4_OLIVE_OPTIMIZER.md)**
* **[Phase 16: ONNX-Simplifier](ONNX5_ONNX_SIMPLIFIER.md)**
* **[Phase 17: ONNXScript / Spox](ONNX6_ONNXSCRIPT_SPOX.md)**
* **[Phase 18: ORT Native Exec](ONNX7_ORT_NATIVE.md)**
* **[Phase 19: tf2onnx](ONNX8_TF2ONNX.md)**
* **[Phase 20: paddle2onnx](ONNX9_PADDLE2ONNX.md)**
* **[Phase 21: skl2onnx](ONNX10_SKL2ONNX.md)**
* **[Phase 22: onnxmltools](ONNX11_ONNXMLTOOLS.md)**
* **[Phase 23: ONNX GraphSurgeon](ONNX12_GRAPHSURGEON.md)**
* **[Phase 24: Hummingbird](ONNX13_HUMMINGBIRD.md)**
* **[Phase 25: Netron](ONNX14_NETRON.md)**
* **[Phase 26: onnx-tool](ONNX15_ONNX_TOOL.md)**
* **[Phase 27: onnx-mlir](ONNX16_ONNXMLIR.md)**
* **[Phase 28: onnx-safetensors](ONNX17_SAFETENSORS.md)**
