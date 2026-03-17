# ONNX9000 Roadmap

This document outlines the current state and future milestones of the `onnx9000` ecosystem. The roadmap is divided into architectural refactoring phases and feature-specific implementation specifications.

## 🚀 Current Status: Foundation Complete

We have successfully executed the **Polyglot Monorepo Refactor**.
The massive single-directory Python monolith has been cleanly split into a highly modular, decoupled environment managed by `pnpm` and `uv` workspaces.

- **Python Core:** The `onnx9000-core` package now parses `.onnx` and `.safetensors` files with zero external dependencies using `struct` and `mmap` directly to an AST.
- **Python EPs:** `onnx9000-backend-native` provides `ctypes` bindings to OpenBLAS/Accelerate, mapping our custom Tensors via DLPack interfaces.
- **TypeScript Core:** `@onnx9000/core` implements an exact structural clone of the ONNX AST with the strictest possible type safety (no `any`, `unknown`).

## 🗺️ Implementation Specifications (The 31 Specs)

The following architectural targets guide the development of the ecosystem. They are grouped by their respective domains.

### Core Execution & Web Backends

- [x] **ONNX00:** Runtime (Native Exec) Replication & Parity Tracker (The `onnx9000-backend-native` package).
- [ ] **ONNX01:** ONNX Standard Compliance & Testing Tracker.
- [ ] **ONNX03:** ONNX Runtime Web Replication (`@onnx9000/backend-web`).
- [ ] **ONNX25:** WebNN API Native Browser NPU Execution.

### Tooling, Parsing, and Optimizations

- [ ] **ONNX04:** ONNX Runtime Extensions Replication (`onnx9000-frontend` & `@onnx9000/transformers`).
- [ ] **ONNX06:** Olive Optimizer Replication (Quantization and W4A16 targeting in `onnx9000-optimizer`).
- [ ] **ONNX07:** ONNX Simplifier Replication (AST Rewriting in `onnx9000-optimizer`).
- [ ] **ONNX14:** ONNX GraphSurgeon Replication.
- [ ] **ONNX17:** `onnx-tool` Profiling Replication (MACs/FLOPs extraction).
- [ ] **ONNX22:** Safetensors Replication (Zero-copy `mmap` and `ArrayBuffer` extraction in `onnx9000-toolkit`).

### Frontends & Converters (`onnx9000-frontends`)

- [ ] **ONNX05:** Torch & TF Exporters Replication.
- [ ] **ONNX10:** `tf2onnx` Replication (Zero-dependency TF parsing).
- [ ] **ONNX11:** `paddle2onnx` Replication.
- [ ] **ONNX12:** `skl2onnx` Replication (Compiling Scikit-Learn to `ai.onnx.ml`).
- [ ] **ONNX13:** `onnxmltools` Replication (LightGBM, XGBoost to ONNX).
- [ ] **ONNX15:** Hummingbird Replication (Compiling Trees to Tensor Math).
- [ ] **ONNX28:** `keras2onnx` & `tfjs-to-onnx` (Web-Native Keras Converter).
- [ ] **ONNX31:** `MMdnn` (Web-Native N-to-N Neural Network Converter).

### Compilers & AOT (`@onnx9000/compiler`)

- [ ] **ONNX19:** `onnx-mlir` Replication (Compiling ONNX to C++23/WASM).
- [ ] **ONNX20:** Apache TVM Ahead-of-Time Web Compiler.
- [ ] **ONNX26:** Apache TVM IREE (WASM-Native MLIR Compiler).
- [ ] **ONNX27:** `coremltools` (Web-Native Apple Silicon Bridge).

### Web UI & Applications (`apps/`)

- [ ] **ONNX16:** Netron Replication (The Vanilla TS WebGL Visualizer `netron-ui`).
- [ ] **ONNX24:** HuggingFace Optimum (Web-Optimized Export & Quantization UI).
- [ ] **ONNX29:** `onnx-modifier` (Web-Native Graph Editor & Visualizer).

### High-Level APIs & GenAI

- [ ] **ONNX02:** ONNX Runtime Training Replication (AOT Symbolic Autograd).
- [ ] **ONNX08:** ONNXScript / Spox Replication (Fluent Model Authoring).
- [ ] **ONNX21:** ONNX Runtime GenAI (WASM-First Generative Execution).
- [ ] **ONNX23:** Transformers.js (WASM-Native Auto-Pipelines).
- [ ] **ONNX30:** `onnx-array-api` (Web-Native NumPy/Eager API for ONNX).

## 🔮 The "Next Next" Plan

Once the core specifications are complete, the `onnx9000` ecosystem will expand into a **Distributed MLOps Framework** as detailed in `ONNX_NEXT_NEXT_PLAN.md`. This will include:

1. P2P WebRTC tensor data channels bridging browser nodes.
2. Distributed Pipeline Parallelism mapping Subgraphs across disparate devices.
3. A zero-dependency Python MLOps server leveraging an embedded SQLite database for tracking metrics and models across the cluster.
