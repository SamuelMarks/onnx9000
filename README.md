<!-- BADGES -->
![Doc Coverage](https://img.shields.io/badge/Doc_Coverage-74%25-blue) ![Test Coverage](https://img.shields.io/badge/Test_Coverage-100%25-success)
<!-- /BADGES -->

# ONNX9000 🚀

> **The Zero-Dependency, WASM-First, Polyglot Machine Learning Ecosystem.**

`onnx9000` is a radical reimagining of the Machine Learning deployment stack. It completely drops massive C++ binaries, complex CMake toolchains, and bloated Python dependencies (like `numpy`, `torch`, or `tensorflow`) in favor of a **Polyglot Monorepo** written in pure Python and strictly-typed TypeScript.

The goal is absolute portability: an ONNX model should parse, optimize, compile, and execute flawlessly on a massive GPU cluster, a Serverless Node.js function, or directly inside a user's web browser using WebAssembly and WebGPU—all with zero installation overhead.

## The Polyglot Monorepo Architecture

`onnx9000` is divided into highly cohesive, modular packages ensuring that the core Intermediate Representation (IR) is completely decoupled from Execution Providers (EPs) and Frontends.

### 🐍 Python Packages

- `packages/python/onnx9000-core`: The zero-dependency ONNX Protobuf parser, AST (Abstract Syntax Tree), and static shape inference engine.
- `packages/python/onnx9000-backend-native`: The `InferenceSession` orchestrator and hardware execution providers (CUDA, Apple Accelerate, CPU, TensorRT) utilizing `ctypes` and zero-copy memory arenas instead of heavy `numpy` arrays.
- `packages/python/onnx9000-optimizer`: The optimization toolkit containing GraphSurgeon, Simplifier, Olive Quantization (INT8, W4A16), and `onnx-tool` profiling logic.
- `packages/python/onnx9000-converters`: Translators for legacy frameworks (PyTorch, TensorFlow, Scikit-Learn, Keras, PaddlePaddle) exporting directly to the pure `onnx9000` AST.
- `packages/python/onnx9000-toolkit`: Eager execution APIs, ONNXScript, Autograd (VJP generators), and the `.safetensors` zero-copy memory-mapped reader.

### 🌐 TypeScript Packages

- `packages/js/core`: Pure TypeScript, strictly-typed implementation of the ONNX AST (no `any`, `unknown`, or `never` allowed), fully isomorphic for Node.js, Deno, Bun, and the Browser.
- `packages/js/backend-web`: WebGPU, WebNN, and WebAssembly (WASM SIMD) execution providers mapped to the TS AST.
- `packages/js/compiler`: Ahead-Of-Time (AOT) transpilers converting MLIR dialects directly to WGSL shaders and WASM bytecodes (`.wvm`).
- `packages/js/transformers`: A Web-Native implementation of `transformers.js` and `diffusers`, supplying BPE tokenizers, pipelines, and schedulers entirely via WASM.
- `packages/js/serve`: Edge-native inference server providing KServe V2 and OpenAI REST APIs targeting Cloudflare Workers and Bun.
- `packages/js/tfjs-shim`: A 100% drop-in API replacement for TensorFlow.js, transparently routing apps to the WebGPU ONNX execution engine.

### 📱 Applications (Vanilla Web)

- `apps/cli`: The unified Python CLI (`onnx9000`) for inspecting, optimizing, and compiling models.
- `apps/netron-ui`: A completely client-side Vanilla TypeScript + WebGL visualizer and graph editor (no React, no Angular, no server backend).
- `apps/optimum-ui`: A web dashboard for visually applying graph fusions, INT4 quantizations, and layout optimizations to ONNX payloads.

## Features & Roadmap Status

- **100% Zero-Dependency Parsers**: Reads `.onnx`, `.pb`, `.h5`, and `.safetensors` natively in Python (`struct`/`mmap`) and JS (`ArrayBuffer`/`DataView`).
- **Static Memory Arenas**: The execution providers completely eliminate dynamic memory allocations (`malloc`/`new`) during inference. Offsets are pre-calculated topologically.
- **WebGPU First**: The ecosystem assumes the final execution target is likely a constrained WebGPU/WebNN browser environment, optimizing layouts (NHWC) and padding specifically for shader compatibility.
- **Bi-Directional Transpilation**: Converts ONNX models flawlessly into TFLite FlatBuffers, CoreML archives, GGUF binaries, and OpenVINO XML structures using zero-dependency AST compilation.
- **Modern Codegen**: When targeting native executables or Emscripten, `onnx9000` emits safe, exception-free C++23, C99 for TinyML microcontrollers, or `Triton` Python kernels for Nvidia GPUs.

## Getting Started

For a comprehensive dive into using the Python API, Web APIs, and CLI tools, please see [USAGE.md](./USAGE.md).

For the complete architectural layout and contribution guidelines, review [ARCHITECTURE.md](./ARCHITECTURE.md) and [ROADMAP.md](./ROADMAP.md).

## Replaced Ecosystem Components

The following table tracks the complete reimplementation and replacement of major first-party and third-party tools within the ONNX and general Machine Learning ecosystem by `onnx9000`.

| Component / Original Project                                                                                                            | Description                                                                          | Tasks Completed | Status     |
| :-------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- | :-------------- | :--------- |
| **ONNX Runtime**<br>[Original](https://github.com/microsoft/onnxruntime) • [Tasks](./specs/ONNX00_RUNTIME.md)                           | Core execution engine for evaluating ONNX models dynamically.                        | 317/317 | ✅ Full |
| **ONNX Compliance**<br>[Original](https://github.com/onnx/onnx) • [Tasks](./specs/ONNX01_COMPLIANCE.md)                                 | Standard testing suite validating correct ONNX mathematical evaluation.              | 327/327 | ✅ Full |
| **ORT Training**<br>[Original](https://github.com/microsoft/onnxruntime) • [Tasks](./specs/ONNX02_ORT_TRAINING.md)                      | Provides Automatic Differentiation (Autograd) and gradient tracking for ONNX models. | 175/303 | 🚧 Partial |
| **ONNX Runtime Web**<br>[Original](https://github.com/microsoft/onnxruntime) • [Tasks](./specs/ONNX03_ORT_WEB.md)                       | In-browser execution engine relying on WebAssembly and WebGPU.                       | 313/313 | ✅ Full |
| **ORT Extensions**<br>[Original](https://github.com/microsoft/onnxruntime-extensions) • [Tasks](./specs/ONNX04_ORT_EXTENSIONS.md)       | Custom operators for text tokenization, audio extraction, and image processing.      | 310/310 | ✅ Full |
| **torch.onnx**<br>[Original](https://pytorch.org) • [Tasks](./specs/ONNX05_TORCH_EXPORTERS.md)                                          | PyTorch to ONNX graph translation tools.                                             | 326/326 | ✅ Full |
| **Olive Optimizer**<br>[Original](https://github.com/microsoft/Olive) • [Tasks](./specs/ONNX06_OLIVE_OPTIMIZER.md)                      | Hardware-aware model optimization, compression, and quantization framework.          | 310/310 | ✅ Full |
| **ONNX Simplifier**<br>[Original](https://github.com/daquexian/onnx-simplifier) • [Tasks](./specs/ONNX07_ONNX_SIMPLIFIER.md)            | A tool to simplify ONNX models by constant folding and algebraic rewriting.          | 108/310 | 🚧 Partial |
| **ONNXScript / Spox**<br>[Original](https://github.com/microsoft/onnxscript) • [Tasks](./specs/ONNX08_ONNXSCRIPT_SPOX.md)               | Authoring ONNX graphs directly via a PyTorch-like Python API.                        | 306/306 | ✅ Full |
| **ORT Native EP**<br>[Original](https://github.com/microsoft/onnxruntime) • [Tasks](./specs/ONNX09_ORT_NATIVE.md)                       | Native hardware execution providers (CUDA, CoreML, DirectML).                        | 313/313 | ✅ Full |
| **tf2onnx**<br>[Original](https://github.com/onnx/tensorflow-onnx) • [Tasks](./specs/ONNX10_TF2ONNX.md)                                 | Converts TensorFlow (SavedModel/GraphDef) to ONNX format.                            | 340/340 | ✅ Full |
| **paddle2onnx**<br>[Original](https://github.com/PaddlePaddle/Paddle2ONNX) • [Tasks](./specs/ONNX11_PADDLE2ONNX.md)                     | Converts PaddlePaddle models to ONNX format.                                         | 324/324 | ✅ Full |
| **skl2onnx**<br>[Original](https://github.com/onnx/sklearn-onnx) • [Tasks](./specs/ONNX12_SKL2ONNX.md)                                  | Translates Scikit-Learn models into ONNX ML (ai.onnx.ml) topologies.                 | 311/311 | ✅ Full |
| **onnxmltools**<br>[Original](https://github.com/onnx/onnxmltools) • [Tasks](./specs/ONNX13_ONNXMLTOOLS.md)                             | Translates LightGBM, XGBoost, CatBoost, and SparkML to ONNX ML.                      | 307/307 | ✅ Full |
| **GraphSurgeon**<br>[Original](https://github.com/NVIDIA/TensorRT) • [Tasks](./specs/ONNX14_GRAPHSURGEON.md)                            | Allows surgical modification, pruning, and subgraph extraction of ONNX files.        | 303/303 | ✅ Full |
| **Hummingbird**<br>[Original](https://github.com/microsoft/hummingbird) • [Tasks](./specs/ONNX15_HUMMINGBIRD.md)                        | Transpiles traditional ML models (Trees) into pure tensor math for GPU acceleration. | 320/320 | ✅ Full |
| **Netron**<br>[Original](https://github.com/lutzroeder/netron) • [Tasks](./specs/ONNX16_NETRON.md)                                      | Visualizes deep learning and machine learning model topologies.                      | 304/304 | ✅ Full |
| **onnx-tool**<br>[Original](https://github.com/aiedu-research/onnx-tool) • [Tasks](./specs/ONNX17_ONNX_TOOL.md)                         | Diagnostic utility for profiling MACs, FLOPs, and static memory footprint.           | 0/306 | ⏳ TODO |
| **onnx-mlir**<br>[Original](https://github.com/onnx/onnx-mlir) • [Tasks](./specs/ONNX19_ONNXMLIR.md)                                    | Compiles ONNX models to LLVM/MLIR formats and C++ executables.                       | 0/320 | ⏳ TODO |
| **Apache TVM**<br>[Original](https://github.com/apache/tvm) • [Tasks](./specs/ONNX20_TVM_COMPILER.md)                                   | AOT machine learning compiler framework for highly tuned standalone execution.       | 0/350 | ⏳ TODO |
| **ORT GenAI**<br>[Original](https://github.com/microsoft/onnxruntime-genai) • [Tasks](./specs/ONNX21_ORT_GENAI.md)                      | Specialized execution loops for Generative AI (LLMs, Whisper).                       | 0/300 | ⏳ TODO |
| **safetensors**<br>[Original](https://github.com/huggingface/safetensors) • [Tasks](./specs/ONNX22_SAFETENSORS.md)                      | Zero-copy, secure tensor serialization format by Hugging Face.                       | 0/309 | ⏳ TODO |
| **Transformers.js**<br>[Original](https://github.com/xenova/transformers.js) • [Tasks](./specs/ONNX23_TRANSFORMERS_JS.md)               | Runs Hugging Face transformer models directly in the browser/Node.js.                | 0/300 | ⏳ TODO |
| **Optimum**<br>[Original](https://github.com/huggingface/optimum) • [Tasks](./specs/ONNX24_OPTIMUM.md)                                  | Web-optimized export & quantization (W4A16, GPTQ, AWQ) toolchain.                    | 0/300 | ⏳ TODO |
| **WebNN API**<br>[Original](https://www.w3.org/TR/webnn/) • [Tasks](./specs/ONNX25_WEBNN_EP.md)                                         | Web API for accessing dedicated hardware accelerators (NPU) in the browser.          | 0/300 | ⏳ TODO |
| **OpenXLA IREE**<br>[Original](https://github.com/openxla/iree) • [Tasks](./specs/ONNX26_APACHE_TVM_IREE.md)                            | AOT compilation to standalone VM bytecodes and tiny execution runtimes.              | 0/300 | ⏳ TODO |
| **coremltools**<br>[Original](https://github.com/apple/coremltools) • [Tasks](./specs/ONNX27_COREMLTOOLS.md)                            | Apple's tool for converting ML models into Core ML format.                           | 0/300 | ⏳ TODO |
| **keras2onnx & tfjs**<br>[Original](https://github.com/onnx/keras-onnx) • [Tasks](./specs/ONNX28_KERAS2ONNX.md)                         | Translates Keras (H5/Keras3) and TensorFlow.js models into ONNX.                     | 0/300 | ⏳ TODO |
| **onnx-modifier**<br>[Original](https://github.com/ZhangGe6/onnx-modifier) • [Tasks](./specs/ONNX29_ONNX_MODIFIER.md)                   | Web-based graphical editor for ONNX models.                                          | 0/300 | ⏳ TODO |
| **onnx-array-api**<br>[Original](https://github.com/sdpython/onnx-array-api) • [Tasks](./specs/ONNX30_ONNX_ARRAY_API.md)                | A NumPy-like eager execution API for dynamic ONNX graph construction.                | 0/300 | ⏳ TODO |
| **MMdnn**<br>[Original](https://github.com/Microsoft/MMdnn) • [Tasks](./specs/ONNX31_MMDNN.md)                                          | N-to-N converter between various deep learning frameworks (Caffe, MXNet, etc.).      | 0/300 | ⏳ TODO |
| **onnx2tf (PINTO0309)**<br>[Original](https://github.com/PINTO0309/onnx2tf) • [Tasks](./specs/ONNX32_ONNX2TF.md)                        | Web-Native TFLite & EdgeTPU Exporter.                                                | 0/330 | ⏳ TODO |
| **onnx2c / deepC**<br>[Original](https://github.com/ai-techsystems/deepC) • [Tasks](./specs/ONNX33_ONNX2C.md)                           | Web-Native TinyML & Embedded C99 Generator.                                          | 0/300 | ⏳ TODO |
| **onnx2gguf**<br>[Original](https://github.com/ggerganov/llama.cpp) • [Tasks](./specs/ONNX34_ONNX2GGUF.md)                              | Web-Native GGUF Compiler & Llama.cpp Bridge.                                         | 0/300 | ⏳ TODO |
| **SparseML**<br>[Original](https://github.com/neuralmagic/sparseml) • [Tasks](./specs/ONNX35_SPARSEML.md)                               | Web-Native Sparsity & Pruning Engine.                                                | 0/300 | ⏳ TODO |
| **TF.js API Shim**<br>[Original](https://github.com/tensorflow/tfjs) • [Tasks](./specs/ONNX36_TFJS_SHIM.md)                             | WebGPU ONNX Drop-In Replacement for TF.js.                                           | 0/300 | ⏳ TODO |
| **ONNX-TensorRT**<br>[Original](https://github.com/onnx/onnx-tensorrt) • [Tasks](./specs/ONNX37_TENSORRT.md)                            | Zero-Build TRT FFI Parser.                                                           | 0/300 | ⏳ TODO |
| **Triton Compiler**<br>[Original](https://github.com/openai/triton) • [Tasks](./specs/ONNX38_TRITON.md)                                 | Web-Native Custom Kernel Generator.                                                  | 0/300 | ⏳ TODO |
| **WebNN Polyfill**<br>[Original](https://github.com/webmachinelearning/webnn-polyfill) • [Tasks](./specs/ONNX39_WEBNN_POLYFILL.md)      | W3C API WebGPU/WASM Shim.                                                            | 0/300 | ⏳ TODO |
| **ONNX Checker**<br>[Original](https://github.com/onnx/onnx) • [Tasks](./specs/ONNX40_ONNX_CHECKER.md)                                  | 100% Pure TS/Python Web-Native Schema Validator.                                     | 0/300 | ⏳ TODO |
| **OpenVINO Optimizer**<br>[Original](https://github.com/openvinotoolkit/openvino) • [Tasks](./specs/ONNX41_OPENVINO.md)                 | Zero-dependency OpenVINO IR `.xml`/`.bin` Compiler.                                  | 0/300 | ⏳ TODO |
| **Triton Inference Server**<br>[Original](https://github.com/triton-inference-server/server) • [Tasks](./specs/ONNX42_TRITON_SERVER.md) | Serverless Edge Serving Engine (Bun/Cloudflare).                                     | 0/300 | ⏳ TODO |
| **Diffusers**<br>[Original](https://github.com/huggingface/diffusers) • [Tasks](./specs/ONNX43_DIFFUSERS.md)                            | Web-Native Diffusion Pipelines (SDXL, VAE).                                          | 0/300 | ⏳ TODO |
