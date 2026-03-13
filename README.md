onnx9000
========

[![CI](https://github.com/samuel/onnx9000/actions/workflows/ci.yml/badge.svg)](https://github.com/samuel/onnx9000/actions/workflows/ci.yml)
[![Test Coverage](https://img.shields.io/badge/test%20coverage-100%25-brightgreen.svg)](https://github.com/samuel/onnx9000)
[![Doc Coverage](https://img.shields.io/badge/doc%20coverage-100.0%25-brightgreen.svg)](https://github.com/samuel/onnx9000)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Welcome to **onnx9000**, a fundamentally reimagined execution engine for ONNX (Open Neural Network Exchange). Instead of operating as a heavyweight runtime with thousands of pre-compiled operator kernels and dynamic dispatch logic, `onnx9000` is a **JIT transpiler**. It parses your specific ONNX graph and generates heavily optimized, bespoke C++23 or WebAssembly (`.wasm`) code specifically tailored for *that exact model*.

This project provides zero-dependency protobuf parsing, strict type checking, robust autograd capabilities, and 100% test coverage.

**🔥 STATUS:** `onnx9000` is maturing rapidly. Over 200 standard ONNX operators have been implemented. The core IR parsing, autograd engine, static C++ memory planning, Apple Accelerate framework, WebAssembly SIMD backends, CUDA target, and control flow operators are fully integrated.


## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [How it Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Defining and Tracing a Graph](#defining-and-tracing-a-graph)
  - [Compiling to C++ (JIT)](#compiling-to-c-jit)
  - [Compiling to WebAssembly (WASM)](#compiling-to-webassembly-wasm)
- [Project Documentation](#project-documentation)
- [Contributing](#contributing)
- [License](#license)

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
