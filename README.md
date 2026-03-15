onnx9000: The Grand Unification
===============================

[![CI](https://github.com/samuel/onnx9000/actions/workflows/ci.yml/badge.svg)](https://github.com/samuel/onnx9000/actions/workflows/ci.yml)
[![Test Coverage](https://img.shields.io/badge/test%20coverage-100%25-brightgreen.svg)](https://github.com/samuel/onnx9000)
[![Doc Coverage](https://img.shields.io/badge/doc%20coverage-100.0%25-brightgreen.svg)](https://github.com/samuel/onnx9000)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Welcome to **onnx9000 V2**, a fundamentally reimagined ecosystem for ONNX (Open Neural Network Exchange). 

The standard ONNX ecosystem is heavily fragmented across massive C++ repositories (`onnxruntime`, `onnxruntime-extensions`, `torch.onnx`, `tf2onnx`, `Olive`, `onnx-simplifier`). This fragmentation leads to massive binary bloat (150MB+ runtimes), complex build toolchains (CMake, LLVM, Protobuf), and a severe inability to execute natively and efficiently in constrained environments like web browsers (WASM/WebGPU) or edge devices.

`onnx9000` is solving this by **rebuilding the entire ONNX ecosystem into a single, zero-dependency, pure Python and JS/WASM/WebGPU architecture.**

By utilizing a unified Intermediate Representation (IR) written entirely in Python, we can author, trace, optimize, quantize, and execute models dynamically without ever compiling a heavy C++ runtime. 

## The Reimplementation Master Plan

To achieve true zero-dependency web execution and lightweight native deployments, we are systematically rewriting 18 major ONNX/Ecosystem projects natively into this repository:

| Target Project | `onnx9000` V2 Home | Description | Status |
| :--- | :--- | :--- | :--- |
| **[ONNX Runtime Training](https://github.com/microsoft/onnxruntime/tree/main/orttraining)** <br> 📝 [Checklist](ONNX0_ORT_TRAINING.md) | `src/onnx9000/training/` | Replaces the C++ training runtime by statically generating Ahead-of-Time (AOT) backward passes, losses, and optimizer steps entirely as forward ONNX math ops. | 🟢 Ported |
| **[ONNX Runtime Web](https://github.com/microsoft/onnxruntime/tree/main/js/web)** <br> 📝 [Checklist](ONNX1_ORT_WEB.md) | `src/onnx9000/backends/web/` | Replaces the heavy Emscripten C++ build with a ground-up WebGPU/WASM engine written natively in JS, supporting progressive chunked loading of weights. | 🟢 Ported |
| **[ONNX Runtime Extensions](https://github.com/microsoft/onnxruntime-extensions)** <br> 📝 [Checklist](ONNX2_ORT_EXTENSIONS.md) | `src/onnx9000/extensions/` | Replaces HuggingFace tokenizers (Rust/C++) and FFmpeg with pure Python/JS BPE tokenizers and WebCodecs/WebAudio media loaders. | 🟢 Ported |
| **[Torch/TF Exporters](https://github.com/pytorch/pytorch/tree/main/torch/onnx)** <br> 📝 [Checklist](ONNX3_TORCH_EXPORTERS.md) | `src/onnx9000/frontends/` | Replaces massive `torch.onnx` C++ tracing with a lightweight, pure-Python PyTorch-like API that traces models directly to ONNX IR in Pyodide. | 🟢 Ported |
| **[Olive Optimizer](https://github.com/microsoft/Olive)** <br> 📝 [Checklist](ONNX4_OLIVE_OPTIMIZER.md) | `src/onnx9000/optimize/hardware/` | Replaces Microsoft's heavy C++ hardware optimizer with pure Python INT8/INT4 quantization and memory layout packing designed for HTTP streaming. | 🟢 Ported |
| **[ONNX-Simplifier](https://github.com/daquexian/onnx-simplifier)** <br> 📝 [Checklist](ONNX5_ONNX_SIMPLIFIER.md) | `src/onnx9000/optimize/simplifier/` | Replaces C++ ONNX Runtime constant folding with aggressive pure-Python algebraic simplifications and dead-code elimination. | 🟢 Ported |
| **[ONNXScript / Spox](https://github.com/microsoft/onnxscript)** <br> 📝 [Checklist](ONNX6_ONNXSCRIPT_SPOX.md) | `src/onnx9000/script/` | Replaces `protobuf` C++ dependent authoring tools with a fluent, pure Python API to dynamically construct graphs node-by-node. | 🟢 Ported |
| **[ORT Native Exec](https://github.com/microsoft/onnxruntime)** <br> 📝 [Checklist](ONNX7_ORT_NATIVE.md) | `src/onnx9000/backends/{cuda,apple}` | Replaces the 150MB+ C++ runtime with a zero-overhead Python dispatcher that dynamically calls `cuBLAS`/`Accelerate` via `ctypes`. | 🟢 Ported |
| **[tf2onnx](https://github.com/onnx/tensorflow-onnx)** <br> 📝 [Checklist](ONNX8_TF2ONNX.md) | `src/onnx9000/frontends/tf/` | Replaces heavy native TF with pure client-side conversion of TensorFlow/Keras/TFLite models to ONNX. | 🟢 Ported |
| **[paddle2onnx](https://github.com/PaddlePaddle/Paddle2ONNX)** <br> 📝 [Checklist](ONNX9_PADDLE2ONNX.md) | `src/onnx9000/frontends/paddle/` | Client-side conversion of PaddlePaddle models to ONNX supporting dynamic shapes and custom CV/NLP subgraphs. | 🟢 Ported |
| **[skl2onnx](https://github.com/onnx/sklearn-onnx)** <br> 📝 [Checklist](ONNX10_SKL2ONNX.md) | `src/onnx9000/frontends/sklearn/` | Translates Scikit-Learn pipelines and estimators into optimized `ai.onnx.ml` graph structures natively in WASM. | ⚪️ Planned |
| **[onnxmltools](https://github.com/onnx/onnxmltools)** <br> 📝 [Checklist](ONNX11_ONNXMLTOOLS.md) | `src/onnx9000/frontends/mltools/` | Unifies converters for XGBoost, LightGBM, CatBoost, CoreML, and SparkML into browser-native graph generation. | ⚪️ Planned |
| **[ONNX GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon)** <br> 📝 [Checklist](ONNX12_GRAPHSURGEON.md) | `src/onnx9000/surgeon/` | High-ergonomic python API for surgical DAG inspection, subgraph replacement, and constant folding via Pyodide. | ⚪️ Planned |
| **[Hummingbird](https://github.com/microsoft/hummingbird)** <br> 📝 [Checklist](ONNX13_HUMMINGBIRD.md) | `src/onnx9000/optimize/hummingbird/` | Transpiles traditional ML tree ensembles into dense tensor math (GEMM/PerfectTree) for lightning-fast WebGPU execution. | ⚪️ Planned |
| **[Netron](https://github.com/lutzroeder/netron)** <br> 📝 [Checklist](ONNX14_NETRON.md) | `src/onnx9000/ui/visualizer/` | Integrates interactive, WASM-accelerated 60FPS graph visualization and live editing directly into the client. | ⚪️ Planned |
| **[onnx-tool](https://github.com/aanna0701/onnx-tool)** <br> 📝 [Checklist](ONNX15_ONNX_TOOL.md) | `src/onnx9000/analysis/profiler/` | Provides zero-install, instant model profiling (MACs, FLOPs, Memory) and dynamic symbolic shape inference. | ⚪️ Planned |
| **[onnx-mlir](https://github.com/onnx/onnx-mlir)** <br> 📝 [Checklist](ONNX16_ONNXMLIR.md) | `src/onnx9000/compiler/mlir/` | AOT compilation of ONNX graphs into MLIR dialects and finally standalone highly optimized WebAssembly binaries. | ⚪️ Planned |
| **[onnx-safetensors](https://github.com/huggingface/safetensors)** <br> 📝 [Checklist](ONNX17_SAFETENSORS.md) | `src/onnx9000/core/safetensors/` | Safe, zero-copy, memory-mapped weight loading explicitly designed for massive LLMs in memory-constrained browsers. | ⚪️ Planned |

## Why Rewrite Everything? (The Main Benefits)

1. **Web-Native Execution (Pyodide & WebGPU):** Because everything from tokenization to quantization and execution is written in pure Python or native JS, the entire pipeline can run flawlessly inside the browser without downloading massive Emscripten binaries.
2. **Zero-Bloat Native Deployments:** For server or desktop execution, you do not need `onnxruntime-gpu`. `onnx9000` evaluates the graph in Python and dynamically dispatches math to native system libraries (NVIDIA cuBLAS, Apple Metal/Accelerate) via `ctypes`. This means instant startup times and zero binary bloat.
3. **Extreme Debuggability:** Standard ONNX errors happen deep inside black-box C++ execution providers. In `onnx9000`, shape inference, constant folding, autograd VJP generation, and execution dispatch are 100% Python. You can step through them with a standard `pdb` debugger.
4. **Unified Optimization:** Because authoring (`script/`), exporting (`frontends/`), and optimizing (`optimize/`) share the exact same lightweight `core.ir` objects in memory, there is zero serialization/deserialization overhead when manipulating the graph.

---

## Introduction

Standard machine learning frameworks and ONNX engines (like ONNX Runtime) suffer from inherent bloat. Because they must support every possible operator, data type, and execution provider, their binary sizes easily exceed 10-30MB. This presents massive challenges for Edge AI, embedded devices, and web browser deployment via WebAssembly.

Furthermore, dynamic operator dispatch at runtime introduces microscopic but compounding latencies, preventing optimal cache locality and compiler-level loop unrolling across operator boundaries.

**onnx9000 solves this via Graph-Specific Transpilation.** By parsing the graph ahead of time (or JIT at runtime), `onnx9000` emits a single C++ source file that hardcodes your model's exact execution path. Unused operators are omitted. Memory is statically planned and allocated in a unified arena. The output is a highly localized, deeply optimizable artifact.

For an in-depth exploration of the project's philosophy, read [WHY.md](./WHY.md).

---

## Key Features

1. **JIT Compilation to C++23**: `onnx9000` translates ONNX operations into strict, modern C++23. It heavily utilizes `std::expected` for monadic error handling, guaranteeing exception-free (`noexcept`) kernel execution boundaries.
2. **First-class WASM Generation**: Seamlessly compile ONNX graphs directly to `.wasm` payloads via Emscripten. The generated WebAssembly requires *no external ML runtime*, making it microscopic (often <50KB for simple models) and incredibly fast to load in `pyodide` or a standard web environment.
3. **Pure Python/Protobuf Frontend**: Zero heavy native dependencies for parsing ONNX. `onnx9000` uses the raw Python `protobuf` library to ingest and manipulate graphs, keeping the installation lightweight and highly portable.
4. **Built-in Autograd Engine**: Features a symbolic reverse-mode automatic differentiation (AD) compiler. You can trace a forward graph, compute VJPs (Vector-Jacobian Products), and compile the *backward graph* directly into C++/WASM for on-device training.
5. **Static Memory Arena**: All intermediate tensor memory is calculated during the transpilation phase. The generated C++ allocates a single contiguous memory block (an arena) to handle all intermediate states, completely eliminating dynamic `malloc`/`free` calls during inference.
6. **Native Hardware Acceleration**: Out-of-the-box support for **Apple Accelerate** (`ONNX9000_USE_ACCELERATE=1`), **WebAssembly SIMD** (via `-msimd128`), and preliminary **CUDA** execution (`ONNX9000_USE_CUDA=1`) for targeted GPU workloads.
7. **Robustness Guarantees**: Enforces strict `mypy` typing across the codebase and maintains a flawless 100% line coverage in `pytest`.

---

## How it Works

The lifecycle of a model in `onnx9000` follows these distinct phases:

1. **Frontend / Parsing**: You can either construct a model purely in Python using `onnx9000.Tensor` and the `@onnx9000.jit` tracing decorator, *or* you can parse an existing standard `.onnx` file using `onnx9000.parser`.
2. **Intermediate Representation (IR)**: The graph is converted into a strictly typed, flattened internal IR. Graph optimizations (like constant folding and dead code elimination) are applied here.
3. **Codegen**: The IR is passed to the Jinja2-based template engine (`src/onnx9000/codegen`). Here, explicit C++ implementations for the required ops are emitted. Memory offsets for all tensors are pre-calculated.
4. **JIT Execution / Compilation**: 
   - If targeting `cpp`, the generated file is compiled via Pybind11 using the system's `g++` or `clang++`. The compiled `.so` / `.dylib` is dynamically loaded and executed.
   - If targeting `wasm`, the system shells out to `emcc` to emit `.js` and `.wasm` artifacts suitable for browser environments.

For a deep dive into the internal modules, read [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## Installation

### Prerequisites

To utilize the full JIT compilation pipeline, you must have:
- Python 3.9 or higher.
- A C++23 compatible compiler (e.g., GCC 12+, Clang 15+).
- Emscripten SDK (`emcc`) - *Optional, only required for WASM targets.*
- CMake (for Pybind11 compilation).

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/samuel/onnx9000.git
cd onnx9000

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install the package with development dependencies
pip install -e ".[dev]"
```

---

## Quick Start

### Defining and Tracing a Graph

`onnx9000` offers a PyTorch-like imperative frontend that automatically traces your Python operations into an ONNX-compliant graph.

```python
import onnx9000
from onnx9000 import DType
import numpy as np

# Use the JIT decorator to trace operations
@onnx9000.jit
def simple_mlp(x, weights, bias):
    # Matrix multiplication
    hidden = x @ weights
    # Element-wise addition and activation
    return onnx9000.ops.relu(hidden + bias)

# Define the symbolic signature of your model
x = onnx9000.Tensor(shape=(1, 128), dtype=DType.FLOAT32, name="x")
w = onnx9000.Parameter(shape=(128, 64), dtype=DType.FLOAT32, name="weights")
b = onnx9000.Parameter(shape=(1, 64), dtype=DType.FLOAT32, name="bias")

# Trace the graph
builder = simple_mlp(x, w, b)

# Export to a standard .onnx protobuf file
onnx9000.to_onnx(builder, "mlp.onnx")
```

### Compiling to C++ (JIT)

Once you have an ONNX graph (either generated by `onnx9000` or exported from PyTorch/TensorFlow), you can JIT compile it directly into an executable Python module.

```python
import onnx9000
import numpy as np

# Parse and compile the ONNX graph to a native C++ extension module
model = onnx9000.compile("mlp.onnx", target="cpp")

# Prepare numpy arrays
x_data = np.random.randn(1, 128).astype(np.float32)
w_data = np.random.randn(128, 64).astype(np.float32)
b_data = np.random.randn(1, 64).astype(np.float32)

# Execute the native module (zero-copy when using standard contiguous arrays)
output = model(x_data, w_data, b_data)

print(f"Output shape: {output.shape}")
print(f"Output type: {output.dtype}")
```

### Compiling to WebAssembly (WASM)

To deploy your model to a web environment, compile it to WebAssembly. This generates a standalone module that does not require ONNX Runtime Web.

*Ensure the Emscripten SDK is activated in your terminal (`source emsdk_env.sh`).*

```python
import onnx9000

# This will generate "mlp.js" and "mlp.wasm" in the output directory
onnx9000.compile("mlp.onnx", target="wasm", out_dir="./web_assets")
```

You can also use the bundled command-line interface:
```bash
onnx9000 compile mlp.onnx --target wasm --out ./web_assets
```

---

## Project Documentation

To truly master `onnx9000`, please consult the extensive documentation provided in the repository:

1. **[USAGE.md](./USAGE.md)**: An exhaustive guide on the API, graph building, debugging tools, and advanced compilation flags.
2. **[ARCHITECTURE.md](./ARCHITECTURE.md)**: A deep dive into the codebase structure, the IR design, the C++ codegen template system, and the static memory allocator.
3. **[WHY.md](./WHY.md)**: The underlying philosophy of the project, detailing the engineering trade-offs and motivations for creating a transpiler rather than a traditional interpreter.

---

## Contributing

We welcome contributions! Please ensure that your pull requests adhere to the project's strict quality standards:

1. **100% Test Coverage**: All new code must be fully covered by unit tests in the `tests/` directory.
2. **Type Hinting**: All Python code must pass strict `mypy` checks.
3. **C++23 Standard**: Any modifications to the Jinja2 C++ templates must emit valid, warning-free C++23.
4. **Code Formatting**: We use `ruff` and `black`. Run the linting suite before committing.

```bash
# Run tests
pytest --cov=src tests/

# Run linters
ruff check src tests
mypy src tests
```

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
