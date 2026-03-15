# ONNX9000 Architecture V2: The Grand Unification

This document outlines the directory structure and architectural separation required to support the massive expansion of `onnx9000`. By absorbing the responsibilities of 18 major ONNX ecosystem projects, `onnx9000` transitions from a simple transpiler into a complete, modular, and lightweight Machine Learning infrastructure.

The core philosophy of V2 is **Zero-Dependency IR first**. The core graph representation, authoring, and optimization should require zero external C++ libraries. Heavy dependencies (like NVIDIA cuBLAS or WebGPU) are strictly isolated to `backends/` and are dynamically loaded or generated only when explicitly requested.

## Proposed Directory Structure (`src/onnx9000/`)

```text
src/onnx9000/
├── core/                # The Heart: IR, Types, and Protobuf
│   ├── ir.py            # Graph and Node representations
│   ├── dtypes.py        # Tensor shapes and types
│   ├── onnx_pb2.py      # Lightweight web-friendly Protobuf parser
│   ├── safetensors/     # Replaces: onnx-safetensors (Zero-copy weight loading)
│   └── ops/             # Standard ONNX operator definitions
│
├── script/              # Replaces: ONNXScript / Spox
│   └── ...              # Fluent pure-Python API for dynamic graph authoring
│
├── frontends/           # Replaces: Torch.ONNX / tf2onnx / paddle2onnx / skl2onnx / onnxmltools
│   ├── torch/           # TorchDynamo / TorchScript tracing without PyTorch C++ backend
│   ├── tf/              # TensorFlow / Keras / TFLite graph extraction
│   ├── paddle/          # PaddlePaddle graph extraction
│   ├── jax/             # JAX jaxpr to ONNX9000 IR
│   ├── sklearn/         # Scikit-Learn pipelines to ai.onnx.ml
│   └── mltools/         # XGBoost, LightGBM, CatBoost, CoreML, SparkML to ONNX
│
├── optimize/            # Replaces: ONNX-Simplifier & Olive & Hummingbird
│   ├── simplifier/      # Pure-Python constant folding, DCE, fusion passes
│   ├── hardware/        # Hardware-aware quantization (INT8/INT4), memory layout optimization
│   └── hummingbird/     # Tensor compilation for Tree Ensembles (GEMM/PerfectTree)
│
├── surgeon/             # Replaces: ONNX GraphSurgeon
│   └── ...              # High-ergonomic python API for surgical DAG inspection & editing
│
├── analysis/            # Replaces: onnx-tool
│   └── profiler/        # MACs, FLOPs, Memory estimation, and Symbolic Shape Inference
│
├── compiler/            # Replaces: onnx-mlir
│   └── mlir/            # AOT compilation of ONNX -> MLIR -> WebAssembly
│
├── ui/                  # Replaces: Netron
│   └── visualizer/      # Interactive, WASM-accelerated 60FPS graph visualization & editing
│
├── training/            # Replaces: ONNX Runtime Training
│   ├── autograd/        # VJP rules and gradient math
│   ├── optimizers/      # SGD, AdamW node generation
│   └── aot_builder.py   # Statically generates training graphs for web/native
│
├── extensions/          # Replaces: ONNX Runtime Extensions
│   ├── text/            # Pure Python/JS BPE, SentencePiece Tokenizers
│   ├── vision/          # Image cropping, resizing, normalization (WebCodecs/Canvas/Python)
│   └── audio/           # Mel-spectrogram, Resampling (WebAudio/Python)
│
├── backends/            # The Muscle: Execution Engines (Native & Web)
│   ├── web/             # Replaces: ORT-Web (WebGPU, WASM, JS bindings)
│   ├── cuda/            # Lightweight CUDA codegen / cuBLAS wrapper (Native)
│   ├── rocm/            # AMD ROCm / HIP bindings (Native)
│   ├── apple/           # Metal / Accelerate framework (Native)
│   └── cpu/             # Fallback Python/Minimal C++ engine
│
├── export/              # Packaging & Serialization
│   ├── chunking.py      # Splitting weights for HTTP Range Requests
│   └── bundle.py        # Assembling WASM/JS/Metadata bundles
│
└── cli/                 # Unified Command Line Interface
```

## How This Achieves the Dual Vision

### 1. The "Web-First" Home
By placing `extensions/` (tokenizers) and `optimize/` (quantization) natively in Python, we can compile these directly to WASM/JS via tools like Pyodide or standard transpilation. The `backends/web/` takes the pure IR and emits raw WebGPU WGSL shaders or WASM SIMD, completely dropping the massive Emscripten C++ payload of traditional ORT-Web.

### 2. The Lighter "Native" Deploy
For native deploys (Linux servers, Windows desktop), ONNX runtime is currently a monolithic ~150MB binary because it statically links every operator and fallback. 
In `onnx9000`, the core is pure Python. If a user runs on an NVIDIA GPU, `backends/cuda/` parses the `core/ir` and either JIT-compiles lightweight `.cu` kernels or uses standard `ctypes`/`cffi` to dynamically call `cuBLAS` directly. There is no monolithic C++ runtime engine—just the Python graph driving the native hardware libraries directly, resulting in zero-overhead dispatch.

## Migration Path from V1 to V2
1. Move `autograd/` -> `training/autograd/`
2. Move `passes/` -> `optimize/simplifier/`
3. Move `frontend/` and `jit/` -> `frontends/`
4. Move `codegen/`, `runtime/`, `wasm/` -> `backends/`
5. Group `ir.py`, `dtypes.py`, `onnx_pb2.py`, `ops/`, `parser/` -> `core/`
6. Refactor internal imports across the codebase.
