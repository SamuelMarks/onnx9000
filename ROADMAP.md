# ONNX9000 Roadmap

This document outlines the current state and future milestones of the `onnx9000` ecosystem. The roadmap is divided into architectural refactoring phases and feature-specific implementation specifications.

## 🚀 Current Status: Foundation Complete

We have successfully executed the **Polyglot Monorepo Refactor**.
The massive single-directory Python monolith has been cleanly split into a highly modular, decoupled environment managed by `pnpm` and `uv` workspaces.

- **Python Core:** The `onnx9000-core` package now parses `.onnx`, `.pb`, and `.safetensors` files with zero external dependencies using `struct` and `mmap` directly to an AST.
- **Python EPs:** `onnx9000-backend-native` provides `ctypes` bindings to OpenBLAS/Accelerate, mapping our custom Tensors via DLPack interfaces.
- **TypeScript Core:** `@onnx9000/core` implements an exact structural clone of the ONNX AST with the strictest possible type safety (no `any`, `unknown`).

## 🗺️ Implementation Specifications (The 44 Specs)

The following architectural targets guide the development of the ecosystem. They are grouped by their respective domains.

### Core Execution & Web Backends

- [x] **ONNX00:** Runtime (Native Exec) Replication & Parity Tracker (`onnx9000-backend-native`).
- [x] **ONNX01:** ONNX Standard Compliance & Testing Tracker.
- [x] **ONNX03:** ONNX Runtime Web Replication (`@onnx9000/backend-web`).
- [x] **ONNX09:** ORT Native EP (CUDA, CoreML, DirectML) Replication.
- [ ] **ONNX25:** WebNN API Native Browser NPU Execution.
- [ ] **ONNX39:** WebNN Polyfill (W3C API WebGPU/WASM Shim).

### Tooling, Parsing, and Optimizations

- [x] **ONNX04:** ONNX Runtime Extensions Replication (`onnx9000-core` & `@onnx9000/transformers`).
- [x] **ONNX06:** Olive Optimizer Replication (Quantization and W4A16 targeting in `onnx9000-optimizer`).
- [x] **ONNX07:** ONNX Simplifier Replication (AST Rewriting in `onnx9000-optimizer`).
- [x] **ONNX14:** ONNX GraphSurgeon Replication.
- [x] **ONNX17:** `onnx-tool` Profiling Replication (MACs/FLOPs extraction).
- [ ] **ONNX22:** Safetensors Replication (Zero-copy `mmap` and `ArrayBuffer` extraction in `onnx9000-toolkit`).
- [ ] **ONNX35:** SparseML Replication (Web-Native Sparsity & Pruning Engine).
- [ ] **ONNX40:** ONNX Checker (100% Pure TS/Python Web-Native Schema Validator).

### Frontends & Converters (`onnx9000-converters`)

- [x] **ONNX05:** Torch & TF Exporters Replication.
- [x] **ONNX10:** `tf2onnx` Replication (Zero-dependency TF parsing).
- [x] **ONNX11:** `paddle2onnx` Replication.
- [x] **ONNX12:** `skl2onnx` Replication (Compiling Scikit-Learn to `ai.onnx.ml`).
- [x] **ONNX13:** `onnxmltools` Replication (LightGBM, XGBoost to ONNX).
- [x] **ONNX15:** Hummingbird Replication (Compiling Trees to Tensor Math).
- [x] **ONNX27:** `coremltools` (Web-Native Apple Silicon Bridge).
- [ ] **ONNX28:** `keras2onnx` & `tfjs-to-onnx` (Web-Native Keras Converter).
- [x] **ONNX31:** `MMdnn` (Web-Native N-to-N Neural Network Converter).
- [ ] **ONNX32:** `onnx2tf` (Web-Native TFLite & EdgeTPU Exporter).
- [ ] **ONNX34:** `onnx2gguf` (Web-Native GGUF Compiler & Llama.cpp Bridge).
- [ ] **ONNX36:** TF.js API Shim (WebGPU ONNX Drop-In Replacement for TF.js).
- [ ] **ONNX37:** ONNX-TensorRT (Zero-Build TRT FFI Parser).

### Compilers & AOT (`@onnx9000/compiler`)

- [x] **ONNX19:** `onnx-mlir` Replication (Compiling ONNX to C++23/WASM).
- [ ] **ONNX20:** Apache TVM Ahead-of-Time Web Compiler.
- [ ] **ONNX26:** Apache TVM IREE (WASM-Native MLIR Compiler).
- [x] **ONNX33:** `onnx2c` / `deepC` (Web-Native TinyML & Embedded C++ Generator).
- [ ] **ONNX38:** Triton Compiler (Web-Native Custom Kernel Generator).
- [ ] **ONNX41:** OpenVINO Optimizer (Zero-dependency OpenVINO IR `.xml`/`.bin` Compiler).

### Web UI & Applications (`apps/`)

- [x] **ONNX16:** Netron Replication (The Vanilla TS WebGL Visualizer `netron-ui`).
- [ ] **ONNX24:** HuggingFace Optimum UI (Web-Optimized Export & Quantization UI `optimum-ui`).
- [ ] **ONNX29:** `onnx-modifier` (Web-Native Graph Editor & Visualizer).
- [ ] **ONNX44:** VS Code Machine Learning OS (The Universal Web-Native IDE).

### High-Level APIs & GenAI

- [x] **ONNX02:** ONNX Runtime Training Replication (AOT Symbolic Autograd in `onnx9000-toolkit`).
- [x] **ONNX08:** ONNXScript / Spox Replication (Fluent Model Authoring).
- [ ] **ONNX21:** ONNX Runtime GenAI (WASM-First Generative Execution).
- [ ] **ONNX23:** Transformers.js (WASM-Native Auto-Pipelines).
- [ ] **ONNX30:** `onnx-array-api` (Web-Native NumPy/Eager API for ONNX).
- [ ] **ONNX42:** Triton Inference Server (Serverless Edge Serving Engine for Bun/Cloudflare).
- [ ] **ONNX43:** Diffusers (Web-Native Diffusion Pipelines like SDXL, VAE).

## 🔮 The "Next Next" Plan

Once the core specifications are complete, the `onnx9000` ecosystem will expand into a **Distributed MLOps Framework** as detailed in `ONNX_NEXT_NEXT_PLAN.md`. This will include:

1. P2P WebRTC tensor data channels bridging browser nodes.
2. Distributed Pipeline Parallelism mapping Subgraphs across disparate devices.
3. A zero-dependency Python MLOps server leveraging an embedded SQLite database for tracking metrics and models across the cluster.

## Framework Support Completeness

For a detailed breakdown of our framework support completeness and % compliant metrics, please see [SUPPORTED_PER_FRAMEWORK.md](SUPPORTED_PER_FRAMEWORK.md).
