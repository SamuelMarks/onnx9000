# Why ONNX9000?

*(Presentation Outline / Manifesto)*

## The Problem: The Crisis of Complexity in ML

Machine Learning infrastructure is broken. 

To run a single model today, developers are forced to wrestle with:
- **Gigabytes of C++ Dependencies:** Massive runtimes that take hours to compile.
- **Tangled CMake Configurations:** Build systems that break across OS boundaries and compiler versions.
- **Python Wheel Hell:** Native bindings that frequently crash due to subtle environment mismatches.
- **The "Deployment Gap":** The harsh reality that the environment used to train a model is almost never the environment used to deploy it.

We created `onnx9000` to break this dependency chain. 

**Our thesis:** Heavy C++ runtimes are no longer necessary for state-of-the-art Machine Learning. With 44 out of 45 ecosystem specifications now completely implemented, we have definitively proven this to be true.

---

## The Solution: 7 Pillars of ONNX9000

### 1. Zero-Dependency by Default
**The Goal:** Absolute portability.
We believe the Intermediate Representation (IR) of a neural network shouldn't require installing `numpy`, `torch`, `protobuf`, or `onnx` C++ wrappers.
- **How we do it:** By parsing `.onnx` and `.safetensors` files using native `struct` unpacking in Python and `DataView` in TypeScript.
- **The Result:** Load, inspect, and execute models on _any_ machine with a standard runtime. No native compilers, no hidden shared libraries.

### 2. The Polyglot Monorepo
**The Goal:** Bridging Data Science and Application Engineering.
Historically, connecting these worlds meant wrapping C++ APIs in messy language bindings. `onnx9000` is a true **Polyglot Monorepo**:
- **Python (`onnx9000-*`):** Handles heavy-lifting tasks (legacy framework exports, complex graph surgery, FFI-based hardware execution).
- **TypeScript (`@onnx9000/*`):** Powers native browser execution (WebGPU, WebNN) and modern UI rendering. No overhead from WASM-to-JS serialization for every operation.

### 3. WASM-First & WebGPU-Native
**The Goal:** Unprecedented browser performance.
Traditional runtimes compile a massive engine to WebAssembly and load graphs dynamically, causing massive memory spikes and slow initialization.
- **Our Approach:** `onnx9000` embraces **Ahead-Of-Time (AOT)** compilation. 
- **The Result:** Our `@onnx9000/compiler` transpiles the AST into micro-binaries or WGSL shaders containing _only_ the math required for your specific model.

### 4. Static Memory Arenas
**The Goal:** Matching native C performance in pure Python or JS.
Dynamic memory allocations (`malloc` / `new`) are the enemy of fast inference.
- **Our Approach:** A `MemoryPlanner` calculates the exact lifespan of every tensor AOT, assigning offsets within a single contiguous `MemoryArena`.
- **The Result:** Elimination of garbage collection pauses and memory fragmentation.

### 5. Rescuing Legacy Models
**The Goal:** Future-proofing ML assets.
Thousands of valuable models are trapped in outdated formats (`.caffemodel`, `.h5`, `.pb`) that require un-installable legacy frameworks.
- **Our Approach:** We built a complete converter suite that revitalizes these architectures, converting them into the modern ONNX standard natively.
- **The Result:** Every major legacy format is now supported without needing the original framework installed.

### 6. The Universal IDE *(Our Final Frontier)*
**The Goal:** A web-native Machine Learning OS.
As our final milestone (Spec 44 out of 45), we are actively building the Universal IDE (`onnx9000 ide`).
- **The Vision:** Bring the entire debugging, visualization, and execution lifecycle into a single VS Code-like interface that runs entirely in the browser. 
- **The Result:** Zero local environment setup required to interact with, modify, and run models.

### 7. The Distributed MLOps Future
**The Goal:** Planet-scale AI infrastructure.
A single node has limits. Having mastered the single-node runtime, `onnx9000` is building the foundation for planet-scale **Peer-to-Peer Browser Swarms**. 
- **The Vision:** By implementing WebRTC data channels, we enable distributed inference and federated training natively in the browser.
- **The Result:** Democratizing AI infrastructure without massive centralized server costs.

---

## Conclusion
`onnx9000` isn't just another runtime; it is a fundamental rethinking of how ML models are parsed, optimized, and delivered. By stripping away legacy C++ baggage and embracing native Web and Python primitives, we are making ML deployment universally accessible, highly performant, and finally—simple.