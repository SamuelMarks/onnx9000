# onnx9000: Why does this exist?

This document outlines the philosophical, technical, and engineering motivations behind `onnx9000`. It answers the core question: *Why build another ONNX execution engine when ONNX Runtime, TensorRT, and OpenVINO already exist?*

**🔥 STATUS:** `onnx9000` is maturing rapidly. Over 200 standard ONNX operators have been implemented. The core IR parsing, autograd engine, static C++ memory planning, Apple Accelerate framework, WebAssembly SIMD backends, CUDA target, WebGPU, and advanced WebWorker RPC architectures are fully integrated and verified with 100% test and doc coverage across Python, C++, and TypeScript.


## Table of Contents

1. [The Problem with General-Purpose Runtimes](#the-problem-with-general-purpose-runtimes)
2. [The Transpilation Paradigm](#the-transpilation-paradigm)
3. [The Problem with Python Bindings](#the-problem-with-python-bindings)
4. [Why C++23 and Static memory?](#why-c23-and-static-memory)
5. [The WebAssembly Imperative](#the-webassembly-imperative)
6. [Conclusion](#conclusion)

---

## 1. The Problem with General-Purpose Runtimes

Frameworks like ONNX Runtime (ORT) are marvels of software engineering. They are incredibly robust, highly optimized, and cover an astronomical surface area of operations, data types, and hardware targets. 

However, this general-purpose nature comes with fundamental structural costs:

1. **Massive Binary Bloat:** To execute an ONNX file, a runtime must ship with compiled C++ code for *every possible operator defined in the ONNX specification*. Even if your model only uses `MatMul`, `Add`, and `Relu`, the runtime binary on your disk contains the logic for `NonMaxSuppression`, `DeformConv`, `LSTMs`, and hundreds of others. This pushes runtime binary sizes to 10MB, 20MB, or even 50MB+.
2. **Dynamic Dispatch Overhead:** During inference, standard runtimes walk the graph node-by-node. For each node, they read a string or enum, perform a dynamic switch/dispatch to locate the correct kernel function, validate types at runtime, allocate memory, execute the kernel, and free the memory. While this takes only microseconds, across a graph with thousands of nodes, this overhead compounds heavily.
3. **Obscured Optimization Horizons:** Because the runtime is compiled ahead of time (AOT) generically, the C++ compiler (`gcc`/`clang`) cannot see your specific graph structure. It cannot perform cross-operator loop unrolling, register caching, or aggressive inlining across operator boundaries.

**`onnx9000` asserts that general-purpose execution is the wrong paradigm for deployment.**

---

## 2. The Transpilation Paradigm

Instead of interpreting a graph, `onnx9000` **transpiles** it. 

When you feed an ONNX model to `onnx9000`, it generates a bespoke `.cpp` text file containing the exact, hardcoded sequence of operations required by that specific model, and nothing else.

**The Benefits:**
- **Zero Operator Bloat:** If your model doesn't use convolutions, the C++ code for convolutions simply isn't generated. The resulting binary is microscopic.
- **Eliminated Dispatch:** There is no runtime graph walking. Execution is a single, hardcoded C++ function calling math routines sequentially.
- **Deep Compiler Optimization:** Because `g++` or `clang++` compiles the graph as a unified C++ file, the compiler's optimizer can analyze the entire execution flow. It can inline functions, keep intermediate variables in CPU registers, and vectorize loops across what used to be impenetrable operator boundaries.

---

## 3. The Problem with Python Bindings

Many machine learning frameworks are effectively massive C++ codebases with thin Python wrappers (using Pybind11 or Cython). This means simply `pip install`-ing the framework requires downloading massive pre-compiled wheel files.

`onnx9000` takes a fundamentally different approach to its architecture. The *framework itself* (the parser, the IR, the memory planner, the autograd engine, the codegen engine) is **pure Python**. It relies only on the pure Python `protobuf` package to read ONNX files.

This makes `onnx9000` incredibly lightweight and trivial to install anywhere. The C++ component is only introduced at the very end of the pipeline, strictly as an output artifact that is compiled on the fly. 

---

## 4. Why C++23 and Static memory?

If we are generating C++ code, we must ensure it is the highest quality, safest, and fastest C++ possible.

### The Static Arena
Dynamic memory allocation (`malloc`, `new`, `free`, `delete`) is the enemy of high-performance computing. It causes thread contention, memory fragmentation, and unpredictable latency spikes.

`onnx9000` uses an **AOT Memory Planner**. Because neural network topologies are generally static, `onnx9000` calculates the exact lifespan and byte-size of every tensor during Python transpilation. It generates C++ code that allocates one single block of memory (the Arena) at startup. All tensors are simply pointer offsets into this single block. This guarantees zero fragmentation and absolute deterministic performance.

### C++23 `std::expected`
Traditional error handling in C++ uses exceptions (`throw` / `catch`). This requires the compiler to inject large "unwind tables" into the binary, increasing size and severely restricting the compiler's ability to optimize instruction reordering.

`onnx9000` mandates modern C++23 constructs. It compiles with `-fno-exceptions` and handles all runtime errors (e.g., shape mismatches during dynamic batching) using `std::expected`. This monadic error handling guarantees `noexcept` boundaries, resulting in smaller, tighter, and infinitely safer native binaries.

---

## 5. The WebAssembly Imperative

The web browser is the ultimate deployment target for edge computing. However, deploying ML to the browser currently requires shipping ONNX Runtime Web, which means pushing megabytes of WebAssembly to the client before the model even loads.

Because `onnx9000` is a transpiler, its WebAssembly target (`target='wasm'`) is a paradigm shift. 

It compiles *only your model's code* into WASM using Emscripten. The resulting `.wasm` payloads are routinely under 50KB. Coupled with the zero-allocation Static Arena architecture, these payloads instantiate instantly in the browser and execute with near-native speeds, utterly bypassing the heavy initialization tax of standard ML web runtimes.

---

## 6. Conclusion

`onnx9000` is not meant to replace PyTorch for exploratory research. It is a razor-sharp deployment tool. It exists for engineers who need absolute control over their runtime footprint, who are deploying to heavily constrained edge devices, or who want to squeeze out the absolute maximum performance by letting an optimizing C++ compiler see their entire graph as a single compilation unit.
