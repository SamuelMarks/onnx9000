# Why ONNX9000?

Machine Learning infrastructure has reached a crisis of complexity and bloat.

To run a simple ML model in a browser or on an edge device today, developers are forced to wrestle with gigabytes of C++ dependencies, tangled CMake configurations, and Python wheels that break across OS boundaries. The official `onnxruntime` is a phenomenally capable tool, but its monolithic C++ nature means porting it to WebAssembly or compiling it for a bespoke embedded target is an agonizing process of patching source code and fighting linker errors.

`onnx9000` was created to break this dependency chain.

## 1. Zero Dependency By Default

We believe the Intermediate Representation (IR) of a neural network shouldn't require installing `numpy`, `torch`, `protobuf`, or `onnx` C++ wrappers.

By parsing `.onnx` and `.safetensors` files using native `struct` unpacking in Python, and `DataView` unpacking in TypeScript, `onnx9000` allows you to load, inspect, and modify models on _any_ machine that has a standard Python or Node.js runtime.

## 2. The Polyglot Monorepo

Machine learning touches data scientists (who speak Python) and application developers (who speak TypeScript/JavaScript). Historically, bridging these worlds meant wrapping C++ APIs in messy language bindings.

`onnx9000` is a true **Polyglot Monorepo**. The core AST (Abstract Syntax Tree) is written natively in both Python (`onnx9000-core`) and TypeScript (`@onnx9000/core`).

- Python handles the heavy lifting of exporting models from legacy frameworks (PyTorch, TensorFlow) and applying complex GraphSurgeon / Olive optimizations.
- TypeScript handles the native browser execution (WebGPU, WebNN) and modern UI rendering (Netron-UI) without crossing heavy WASM-to-JS serialization boundaries for compilation tasks.

## 3. Web-First Execution (WASM & WebGPU)

Traditional runtimes compile the _engine_ to WebAssembly and load the _graph_ dynamically at runtime. This causes massive memory spikes in the browser as the graph is parsed, and introduces branching overhead during execution.

`onnx9000` embraces **Ahead-Of-Time (AOT)** compilation:

- In TypeScript: The `@onnx9000/compiler` package transpiles the ONNX AST into pure C++23. When compiled with Emscripten (`emcc`), the resulting `.wasm` file is a micro-binary containing _only_ the math loops required for your specific model.
- It also lowers the AST directly to WebGPU `WGSL` compute shaders or `WebNN` context instructions, completely bypassing a centralized "runtime engine".

## 4. The Memory Arena

To achieve native C-level performance within pure Python or JS execution backends, `onnx9000` eliminates dynamic memory allocations.

During the optimization phase, a `MemoryPlanner` calculates the exact lifespan of every tensor. It assigns byte-offsets within a single contiguous `MemoryArena`. When the model runs, it executes entirely in-place or via pre-allocated sliding windows. This is how `onnx9000` matches native `cblas` or `CUDA` speeds while dispatching entirely from Python via `ctypes`.

## 5. Rescuing Legacy Models

Thousands of incredible models are trapped in outdated formats (`.caffemodel`, `.h5`, `.pb`). The frameworks required to run them (Caffe, MXNet, TF1) no longer compile on modern operating systems.

The `onnx9000-frontend` package provides pure Python/TS parsers that read these legacy binary formats and translate them directly to the ONNX standard, allowing 10-year-old architectures to suddenly run flawlessly on modern WebGPU browsers with zero native installation.
