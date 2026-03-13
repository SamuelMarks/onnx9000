# onnx9000 Roadmap

**🔥 STATUS:** `onnx9000` is maturing rapidly. Over 200 standard ONNX operators have been implemented. The core IR parsing, autograd engine, static C++ memory planning, Apple Accelerate framework, WebAssembly SIMD backends, CUDA target, and control flow operators are fully integrated.


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
