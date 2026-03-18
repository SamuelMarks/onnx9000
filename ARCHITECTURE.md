# onnx9000: Architecture Deep Dive

This document details the internal architectural design of `onnx9000`. It is intended for core contributors, framework engineers, and advanced users who want to understand exactly how `onnx9000` parses, optimizes, and compiles machine learning graphs into bare-metal C++, WebAssembly, and a dozen other framework formats.

> **Note:** The `onnx9000` architecture relies on a strict **Polyglot Monorepo** design. The core IR is decoupled and isolated in `packages/python/onnx9000-core` and `packages/js/core`. Frontends, EPs, and optimizers MUST never cross-contaminate their dependencies.

## Table of Contents

1. [The Polyglot Monorepo Architecture](#the-polyglot-monorepo-architecture)
2. [The Core Intermediate Representation (IR)](#the-core-intermediate-representation-ir)
3. [The Frontend Converters & Parsers](#the-frontend-converters-parsers)
4. [The Backend Exporters (Bi-Directional Transpilation)](#the-backend-exporters-bi-directional-transpilation)
5. [Hardware Native Backends (Python)](#hardware-native-backends-python)
6. [Web Backends (TypeScript)](#web-backends-typescript)
7. [The Codegen Engine (C++23)](#the-codegen-engine-c-23-triton)
8. [The Autograd Subsystem](#the-autograd-subsystem)
9. [Edge Serving & API Shims](#edge-serving-api-shims)
10. [Tooling, Visualization & Profiling](#tooling-visualization-profiling)

---

(the-polyglot-monorepo-architecture)=

## 1. The Polyglot Monorepo Architecture

`onnx9000` is built as a highly decoupled pipeline. A graph flows through the system in strict, immutable packages managed by `pnpm` workspaces (JS) and `uv` (Python):

```text
graph TD;
    A[Frontends: Torch, TF, Keras, Caffe] -->|Parses to| C(onnx9000-core IR AST);
    B[Raw .onnx / .safetensors] -->|Parses to| C;
    C --> D[Optimizer: Surgeon, Simplifier, SparseML];
    D --> E{Execution Layer};
    E -->|Native| F[Native: CUDA, Accelerate, TensorRT FFI];
    E -->|Web| G[Web: WebGPU, WebNN, WASM];
    E -->|AOT| H[Compiler: IREE, C++ Codegen, Triton];
    D --> I{Exporters};
    I -->|Mobile| J[TFLite, CoreML];
    I -->|LLMs| K[GGUF];
    I -->|Code| L[PyTorch Source, TF.js Source];
```

By decoupling the IR from both the frontend input and the backend output, the system allows arbitrary manipulation (like Autograd or fusion optimizations) strictly within the pure Python/TS domains before a single line of native or shader code is emitted.

---

(the-core-intermediate-representation-ir)=

## 2. The Core Intermediate Representation (IR)

The foundation of the entire ecosystem lives in `packages/python/onnx9000-core` and `packages/js/core`.

The IR (`Graph`, `Node`, `Tensor`, `ValueInfo`) is a flattened, heavily typed list of operations. Unlike the highly nested structure of raw ONNX protobufs, the IR guarantees:

1. **Topological Sorting:** Nodes are strictly ordered so that inputs are always computed before they are consumed.
2. **Shape and Type Inference:** Every single edge in the DAG has a concretely resolved `shape` and `dtype`. Dynamic shapes are handled by preserving symbolic string variables in the shape tuples (e.g., `("N", 64, 64)`).
3. **Zero Dependencies:** The Python package uses native `struct` and `mmap` modules. The TypeScript package uses standard `DataView` and `ArrayBuffer`. Neither relies on `numpy`, `torch`, `onnx`, or proprietary native libraries.
4. **Validation:** Built-in exact parity with `onnx.checker` ensures structural integrity.

---

(the-frontend-converters-parsers)=

## 3. The Frontend Converters & Parsers

The system ingests models through `packages/python/onnx9000-converterss`:

To maintain a zero-heavy-dependency footprint, `onnx9000` avoids using the official C++ `onnx` repository bindings. Instead, it ships with pure Python/TypeScript definitions compiled directly from the ONNX protocol buffers.

- The parsers handle unrolling `GraphProto` objects, decoding base64/raw tensor data for initializers, and resolving legacy layouts (e.g., translating Keras NHWC formats dynamically to ONNX NCHW standard).
- **Extensibility:** Frontends for PyTorch (via Dynamo), Scikit-Learn, XGBoost, and legacy formats (Caffe, MXNet, PaddlePaddle) map their proprietary ASTs directly into `onnx9000-core/ir.py` components natively in the browser.

---

(the-backend-exporters-bi-directional-transpilation)=

## 4. The Backend Exporters (Bi-Directional Transpilation)

Because `onnx9000` treats ONNX as the universal source of truth, it acts as a universal N-to-N converter.

- **TFLite (`onnx2tf`):** Emits FlatBuffers natively to target Android NNAPI and Coral EdgeTPU.
- **CoreML (`coremltools`):** Emits Apple MIL and `.mlpackage` archives targeting the Apple Neural Engine.
- **GGUF (`onnx2gguf`):** Directly translates standard LLM ONNX models into `llama.cpp` compatible `Q4_0` binaries.
- **OpenVINO:** Generates Intel-compatible `.xml` topological trees.
- **Source Code (`MMdnn`):** Uniquely, `onnx9000` can act as an inverse-compiler, generating raw PyTorch `nn.Module` Python files or TF.js JS files representing the imported mathematical topology.

---

(hardware-native-backends-python)=

## 5. Hardware Native Backends (Python)

Living in `packages/python/onnx9000-backend-native`, execution providers are dynamically routed.

- **Static Memory Arenas:** The most profound architectural choice is the total elimination of dynamic memory allocations during inference. Offsets are pre-calculated by the `MemoryPlanner`.
- **CTYPES / FFI Dispatch:** `onnx9000` uses `ctypes` to invoke `cblas_sgemm` (Apple Accelerate), `cublasSgemm` (CUDA), or `nvinfer` (TensorRT) endpoints directly on raw memory. This achieves C++ inference speeds completely within Python.

---

(web-backends-typescript)=

## 6. Web Backends (TypeScript)

Living in `packages/js/backend-web`, execution is ported natively to the browser.

- **WebGPU Shaders (WGSL):** Graph nodes (MatMul, Conv) are compiled dynamically into WebGPU Compute pipelines sharing the exact same memory buffers.
- **WebNN API:** Integrates directly with `navigator.ml` and provides the definitive W3C `webnn-polyfill`.
- **WASM SIMD:** Fallback execution providing high-speed CPU paths.
- **Diffusers & Transformers:** Incorporates complete Web-Native Hugging Face pipelines for generative AI (Stable Diffusion, Whisper, Llama).

---

(the-codegen-engine-c-23-triton)=

## 7. The Codegen Engine (C++23 & Triton)

For targets lacking generic engines:

- **C++ / TinyML:** Maps the Python IR into valid C++23 source code (or C99 for microcontrollers) using Jinja templates, returning `std::expected` or raw pointer arrays.
- **OpenAI Triton:** Identifies un-fusable subgraphs and generates optimized `@triton.jit` Python kernels for execution on Nvidia GPUs.
- **Web-MLIR (IREE):** Compiles static graphs into a minuscule (<50kb) bytecode format (`.wvm`) interpreted entirely in WASM.

---

(the-autograd-subsystem)=

## 8. The Autograd Subsystem

Found in `packages/python/onnx9000-toolkit`, this module performs graph-level Reverse-Mode Automatic Differentiation.

- **AOT Compilation:** It walks the `onnx9000.core.ir` DAG backwards. For every node, it inserts the corresponding VJP (Vector-Jacobian Product) nodes into the graph.
- **Unified Graph:** The result is a single `.onnx` graph containing both the forward inference and backward optimizer steps, executable natively on WebGPU.

---

(edge-serving-api-shims)=

## 9. Edge Serving & API Shims

- **Triton Inference Server:** `onnx9000.serve` provides a high-performance, purely asynchronous server designed for Cloudflare Workers and Bun. It natively implements KServe V2 dynamic batching and OpenAI REST APIs.
- **TF.js Drop-in:** Provides `@onnx9000/tfjs-shim`, allowing developers to replace `@tensorflow/tfjs` entirely. It intercepts `tf.matMul` or `tf.loadGraphModel` calls and routes them transparently to `onnx9000` WebGPU, providing instantaneous acceleration without code rewrites.

(tooling-visualization-profiling)=

## 10. Tooling, Visualization & Profiling

`onnx9000` provides a comprehensive suite of Web-Native visual and diagnostic tools:

- **Graph Editing & Visualization:** A WebGL-accelerated implementation of `Netron` and `onnx-modifier` allows users to open, inspect, and surgically edit massive (>10GB) models directly in the browser at 60FPS. Changes to the topology invoke real-time shape inference and validation before the modified `.onnx` is exported.
- **Diagnostics & Profiling:** A complete port of `onnx-tool` evaluates MACs, FLOPs, and static memory footprint dynamically, giving engineers deep insights into computation bottlenecks without needing to execute the graph.
