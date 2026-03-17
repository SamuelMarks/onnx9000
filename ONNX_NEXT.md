# ONNX NEXT: The Future of ONNX9000

The current Polyglot Monorepo establishes the `onnx9000` execution backend and AST compiler as a highly stable, completely decoupled foundation. The immediate future (The "Next" plan) is entirely focused on mapping the massive breadth of existing ML operators and sub-graphs into this environment, specifically optimizing for the Web.

## The WebGPU / WebNN Imperative

The web is the ultimate target. Current AI frameworks treat the browser as a secondary citizen, relying on heavy C++ middleware (`onnxruntime-web`) to marshal data back and forth from JavaScript.

`onnx9000` fundamentally shifts this paradigm:

1.  **Pure TypeScript AST Evaluation:** Because the AST parser is written in pure TypeScript, compiling dynamic shader strings (`WGSL`) based on node attributes happens instantly in V8/JavaScriptCore, rather than incurring a WebAssembly context-switch penalty.
2.  **Zero-Copy Execution:** The `@onnx9000/backend-web` module implements the Memory Planner directly in JS. All tensors are slices of a single massive `GPUBuffer`.
3.  **W4A16 Quantization:** `onnx9000-optimizer` is heavily focused on implementing specific W4A16 (4-bit weights, 16-bit activations) packing algorithms designed explicitly for how WebGPU handles unaligned reads.

## Imminent Architectural Milestones

- **Generative AI Native Loops:** The `@onnx9000/transformers` module will handle the autoregressive loop (token prediction, KV-Cache management, and Top-K sampling) entirely natively within the WebGPU execution boundaries, removing the JS event-loop overhead from token generation.
- **AOT Web Compiler (IREE/TVM Parity):** Building a `.wvm` (Web Virtual Machine) Bytecode Emitter in `@onnx9000/compiler`. This allows pre-compiled models to be shipped as microscopic 10KB JavaScript payloads containing just the execution queue and WGSL strings, completely bypassing the AST parser at runtime.
- **Netron-UI Edit Mode:** Elevating the `apps/netron-ui` application from a static visualizer into a full `onnx-modifier` and `optimum-ui` implementation. Users will drop an `.onnx` file into the browser, delete nodes, change batch sizes, and apply INT8 quantization visually, downloading the optimized payload instantly without pinging a server.
