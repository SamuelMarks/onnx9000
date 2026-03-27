# The ONNX9000 Ecosystem

To tame the staggering complexity of the modern Machine Learning landscape, `onnx9000` is designed as a modular **Polyglot Monorepo**. Rather than installing completely separate native libraries for inference, parsing, quantization, and UI visualization, everything is tightly integrated around a central, pure-language Intermediate Representation (IR).

By organizing these disparate ML domains into explicit packages, `onnx9000` offers unprecedented web-native compilation speeds while preventing dependency hell.

## Ecosystem Packages

The workspace is cleanly divided between Python tools for heavy-duty graph surgery/exporting and TypeScript web tools for pure execution.

### 🐍 Python Workspace (`packages/python/*`)

1. **`onnx9000-core`**: The foundational package containing the ONNX AST (`Graph`, `Node`, `Tensor`), static shape inference, and zero-dependency Protocol Buffer parsing.
2. **`onnx9000-backend-native`**: The execution layer. Routes nodes topologically to `ExecutionProviders` (CUDA, Apple Accelerate, CPU) utilizing `ctypes` over strict static memory arenas.
3. **`onnx9000-optimizer`**: Includes `GraphSurgeon` for manual topology edits, the `Simplifier` for algebraic constant folding, and `Olive` algorithms for applying INT8 and W4A16 weight quantization. Also extracts FLOP/MAC profiling.
4. **`onnx9000-frontend`**: The translation engines. Translates Scikit-Learn (`skl2onnx`), TensorFlow (`tf2onnx`), PyTorch, LightGBM (`onnxmltools`), Keras, and legacy Caffe/MXNet into valid ONNX structures without needing massive native backend installations.
5. **`onnx9000-toolkit`**: The eager execution APIs (`onnx-array-api`), fluent decorators (`ONNXScript`), and the Ahead-Of-Time (AOT) reverse-mode Autograd engine for generating `.onnx` training graphs.

### 🌐 TypeScript Workspace (`packages/js/*`)

1. **`@onnx9000/core`**: An exact TypeScript clone of the Python core IR. Fully strictly typed, operating flawlessly in Node, Deno, and the Browser.
2. **`@onnx9000/backend-web`**: Implements the `InferenceSession` to dispatch AST nodes dynamically to `WebGPU` compute shaders, the `WebNN` API, or `WASM` SIMD fallbacks.
3. **`@onnx9000/transformers`**: Provides the `pipeline()` API equivalent of `transformers.js`, coupling WASM-accelerated BPE tokenizers and image processors intimately with the execution backend.
4. **`@onnx9000/compiler`**: The Ahead-Of-Time generators simulating MLIR pipelines (IREE / TVM / ONNX-MLIR) to compile dynamic graphs entirely into microscopic standalone `.wasm` or `wgsl` payloads without needing the runtime engine itself.

### 📱 Applications (`apps/*`)

1. **`onnx9000-cli`**: The unified Python orchestrator mapping user terminal commands to the frontend, optimizer, and compilation packages.
2. **`netron-ui`**: A purely vanilla TS implementation of Netron to view, edit, and profile massive `.onnx` models interactively at 60FPS using WebGL.
3. **`optimum-ui`**: A visual dashboard allowing users to execute INT4 quantization techniques on their weights and observe the graph layout changes dynamically before downloading the resulting web-safe payload.

4. **`sphinx-demo-ui`**: The all-encompassing browser IDE embedded in Sphinx documentation, capable of rendering models, and running real-time transpilation to MLIR, C++, PyTorch, Caffe, and Apple CoreML offline.

## Framework Support Completeness

For a detailed breakdown of our framework support completeness and % compliant metrics, please see [SUPPORTED_PER_FRAMEWORK.md](SUPPORTED_PER_FRAMEWORK.md).
