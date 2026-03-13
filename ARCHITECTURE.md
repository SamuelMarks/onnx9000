# onnx9000: Architecture Deep Dive

This document details the internal architectural design of `onnx9000`. It is intended for core contributors, framework engineers, and advanced users who want to understand exactly how `onnx9000` parses, optimizes, and compiles machine learning graphs into bare-metal C++ and WebAssembly.

**🔥 STATUS:** `onnx9000` is maturing rapidly. Over 200 standard ONNX operators have been implemented. The core IR parsing, autograd engine, static C++ memory planning, Apple Accelerate framework, WebAssembly SIMD backends, CUDA target, and control flow operators are fully integrated.


## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [The Frontend & Parser](#the-frontend--parser)
3. [The Intermediate Representation (IR)](#the-intermediate-representation-ir)
4. [The Codegen Engine](#the-codegen-engine)
    - [Static Memory Arena](#static-memory-arena)
    - [C++23 & `std::expected`](#c23--stdexpected)
5. [The Autograd Subsystem](#the-autograd-subsystem)
6. [JIT and Caching Mechanics](#jit-and-caching-mechanics)
7. [WebAssembly Integration](#webassembly-integration)

---

## 1. High-Level Architecture

`onnx9000` is built as a highly decoupled pipeline. A graph flows through the system in strict, immutable stages:

```mermaid
graph TD;
    A[Python `@jit` Frontend] --> C[GraphBuilder AST];
    B[Raw .onnx File] -->|Parser| C;
    C --> D[Internal IR Graph];
    D --> E[Graph Optimization Passes];
    E --> F[Codegen Template Engine];
    F --> G{Target Engine};
    G -->|target='cpp'| H[Pybind11 Wrapper + System C++ Compiler];
    G -->|target='wasm'| I[Embind Wrapper + Emscripten (emcc)];
    H --> J[Executable Shared Object (.so/.dylib)];
    I --> K[WebAssembly Payload (.wasm + .js)];
```

By decoupling the IR from both the frontend input and the backend output, the system allows arbitrary manipulation (like Autograd or fusion optimizations) strictly within the pure Python domain before a single line of native code is emitted.

---

## 2. The Frontend & Parser

The system ingests models through two primary modules:

### `src/onnx9000/frontend/`
This module contains the tracing infrastructure. The `@onnx9000.jit` decorator works by intercepting operator calls. When a `Tensor` or `Parameter` object has an operation applied to it (e.g., `tensor_a + tensor_b`), the frontend does *not* execute mathematical addition. Instead, it records an `Add` node in the `GraphBuilder`.
- `tensor.py`: Defines `Node`, `Tensor`, and `Parameter`.
- `builder.py`: Constructs the directed acyclic graph (DAG) of the traced execution.
- `jit.py`: The context manager and decorator for active tracing.

### `src/onnx9000/parser/`
To maintain a zero-heavy-dependency footprint, `onnx9000` avoids using the official C++ `onnx` repository bindings. Instead, it ships with `onnx_pb2.py` (compiled directly from the ONNX protocol buffer definitions) and manually reconstructs the graph.
- The parser handles unrolling ONNX `GraphProto` objects, decoding base64/raw tensor data for initializers, and resolving topological order.

---

## 3. The Intermediate Representation (IR)

Once ingested, everything is lowered into the `onnx9000` Internal Representation (`src/onnx9000/ir.py`).

The IR is a flattened, heavily typed list of operations. Unlike the highly nested structure of raw ONNX protobufs, the IR guarantees:
1. **Topological Sorting:** Nodes are strictly ordered so that inputs are always computed before they are consumed.
2. **Shape and Type Inference:** Every single edge in the DAG has a concretely resolved `shape` and `dtype`. Dynamic shapes are handled by preserving symbolic string variables in the shape tuples (e.g., `("N", 64, 64)`).
3. **Static Allocation Offsets:** During IR lowering, a "Memory Planner" pass runs. It calculates the byte-size of every tensor and assigns a linear offset (e.g., `tensor_X starts at byte 1024, ends at 4096`). Liveness analysis is performed to reuse memory offsets for tensors whose lifespans do not overlap.

---

## 4. The Codegen Engine

The Codegen engine (`src/onnx9000/codegen/`) is the heart of the transpiler. It uses `Jinja2` templates (`src/onnx9000/templates/`) to map the IR into valid C++23 source code.

### Static Memory Arena
The most profound architectural choice in `onnx9000` is the total elimination of dynamic memory allocations during inference.
- The C++ model object contains a single `std::vector<uint8_t> memory_arena;`
- The size of this arena is pre-computed by the memory planner during the Python IR phase.
- When an operator (e.g., a Convolution) needs to write output, it does not call `new` or `malloc`. Instead, the codegen emits:
  ```cpp
  float* out_ptr = reinterpret_cast<float*>(memory_arena.data() + OFFSET_OUT);
  ```
- This ensures maximum cache locality, zero memory fragmentation, and deterministic execution time.

### C++23 & `std::expected`
`onnx9000` relies strictly on modern C++ paradigms. Traditional ML frameworks throw exceptions when shape mismatches occur, which creates massive binary bloat due to exception unwinding tables.
- `onnx9000` compiles kernels with `-fno-exceptions`.
- Every generated operation returns a `std::expected<void, Error>`.
- Errors are propagated monadically. This makes the generated C++ extremely safe, highly optimizable, and perfectly suited for the rigid WASM environment where cross-language exception unwinding is notoriously fragile.

### Hardware Native Backends
The Codegen engine conditionally injects backend-specific compiler intrinsics or library calls based on the environment configuration:
- **Apple Accelerate:** When `ONNX9000_USE_ACCELERATE=1` is active, loops in math operations (like `Add` or `MatMul`) are replaced with direct calls to macOS's `vDSP` and `vForce` libraries.
- **WASM SIMD:** When targeting WASM, Emscripten-specific `#pragma clang loop vectorize(enable)` pragmas are injected alongside `-msimd128` flags to leverage 128-bit vectorization in the browser.
- **CUDA:** Setting `ONNX9000_USE_CUDA=1` swaps the CPU memory vectors for a `CudaBuffer` wrapper and enables GPU-accelerated path execution for supporting operators.

---

## 5. The Autograd Subsystem

Found in `src/onnx9000/autograd/`, this module performs graph-level Reverse-Mode Automatic Differentiation.

Unlike standard PyTorch which builds an implicit tape at runtime, `onnx9000`'s autograd is a static compiler pass.
1. **VJP Rules:** `rules.py` defines Vector-Jacobian Product rules for every supported ONNX operator.
2. **Reverse Traversal:** `vjp.py` walks the IR DAG backwards. For every node, it inserts the corresponding gradient-calculating nodes into the graph.
3. **Graph Output:** The result is a single, unified forward+backward graph.
This massive joint graph is then passed to the Codegen engine, memory planned, and compiled. This means your training step benefits from the exact same Static Memory Arena and kernel fusions as your inference step.

---

## 6. JIT and Caching Mechanics

When `onnx9000.compile(target="cpp")` is invoked, the JIT engine (`src/onnx9000/jit/`) orchestrates the build.

1. **Hashing:** `hasher.py` computes an MD5 hash of the emitted C++ string, the system architecture, and the compiler version.
2. **Caching:** It checks `~/.cache/onnx9000/` for a `.so` matching this hash. If found, compilation is bypassed entirely, yielding near-instant load times for previously compiled graphs.
3. **Pybind11 Integration:** `wrapper.py` appends Pybind11 `PYBIND11_MODULE` macros to the C++ code, linking Python NumPy arrays directly to the C++ Memory Arena pointers. It uses CMake under the hood to manage cross-platform linking quirks.

---

## 7. WebAssembly Integration

The WASM pipeline (`src/onnx9000/wasm/`) alters the codegen phase. Instead of Pybind11 macros, it appends Emscripten `EMSCRIPTEN_BINDINGS` blocks.

- **Data Bridging:** JS typed arrays (`Float32Array`) are bridged to C++ via `emscripten::val` memory views.
- **Heap Management:** Because WASM executes in a constrained linear memory space, `onnx9000`'s static arena architecture is a perfect fit. The entire model requires exactly one `_malloc` call from JavaScript to the WASM heap to initialize the arena, after which all execution occurs zero-allocation.
- **Microscopic Size:** Because we do not compile a generic runtime, the emitted `.wasm` file contains *only* the math instructions required for your specific nodes. A simple MLP might compile to a WASM payload of under 20 Kilobytes.
