# Why ONNX9000?

Machine Learning infrastructure has reached a crisis of complexity.

To run a model today, developers are forced to wrestle with gigabytes of C++ dependencies, tangled CMake configurations, and Python wheels that break across OS boundaries. `onnx9000` was created to break this dependency chain. With 44 out of 45 feature specifications now completely implemented, we have definitively proven that heavy C++ runtimes are no longer necessary.

## 1. Zero-Dependency by Default

We believe the Intermediate Representation (IR) of a neural network shouldn't require installing `numpy`, `torch`, `protobuf`, or `onnx` C++ wrappers.

By parsing `.onnx` and `.safetensors` files using native `struct` unpacking in Python and `DataView` in TypeScript, `onnx9000` allows you to load, inspect, and execute models on _any_ machine with a standard runtime. No native compilers, no hidden shared libraries.

## 2. The Polyglot Monorepo

Machine learning touches both Data Science (Python) and Application Engineering (TypeScript). Historically, bridging these worlds meant wrapping C++ APIs in messy language bindings.

`onnx9000` is a true **Polyglot Monorepo**:

- **Python (`onnx9000-*`)**: Handles heavy-lifting tasks like exporting from legacy frameworks, complex graph surgery, and FFI-based hardware execution.
- **TypeScript (`@onnx9000/*`)**: Powers native browser execution (WebGPU, WebNN) and modern UI rendering without the overhead of WASM-to-JS serialization for every operation.

## 3. WASM-First & WebGPU-Native

Traditional runtimes compile a massive engine to WebAssembly and load graphs dynamically, causing memory spikes. `onnx9000` embraces **Ahead-Of-Time (AOT)** compilation. Our `@onnx9000/compiler` transpiles the AST into micro-binaries or WGSL shaders containing _only_ the math required for your specific model, achieving unprecedented performance metrics in the browser.

## 4. Static Memory Arenas

To match native C performance, `onnx9000` eliminates dynamic memory allocations. A `MemoryPlanner` calculates the exact lifespan of every tensor AOT, assigning offsets within a single contiguous `MemoryArena`. This enables native execution speeds directly from pure Python or JS.

## 5. Rescuing Legacy Models

Thousands of models are trapped in outdated formats (`.caffemodel`, `.h5`, `.pb`). `onnx9000` revitalizes these architectures, converting them into the ONNX standard without requiring the original, often un-installable, frameworks. With the completion of our converter suite, every major legacy format is now supported.

## 6. The Universal IDE

By building a completely web-native Machine Learning OS (`onnx9000 ide`), we've brought the entire debugging, visualization, and execution lifecycle into a single VS Code-like interface that runs in the browser—no local environment required.

## 7. Distributed MLOps Future

A single node has limits. Having mastered the single-node runtime, `onnx9000` is now building the foundation for planet-scale **Peer-to-Peer Browser Swarms**. By implementing WebRTC data channels, we enable distributed inference and federated training natively in the browser, democratizing AI infrastructure.