---
orphan: true
---

# Demos & Examples for onnx9000-iree

> **Ecosystem Context:** `onnx9000` operates as a zero-dependency, Polyglot Monorepo. Through its integrated Web IDE (`apps/sphinx-demo-ui`), it supports real-time transpilation and offline conversions across C++, PyTorch, MLIR, CoreML, and Caffe targets without native backends.

1. **[Tiny LLM in 20KB JS (241)](https://github.com/onnx9000/tiny-llm-js)**
   A pure Vanilla JS + WebGPU standalone model executing an LLaMA model natively.

2. **[Webcam Object Detection (242)](https://github.com/onnx9000/webcam-yolo-standalone)**
   Real-time YOLOv8 rendering bounding boxes directly into HTML Canvas (243) without any dependencies.

3. **Deno/Bun WVM Execution (244)**
   Example showing how to run `.wvm` bytecode in Deno via `deno run --unstable deno-wvm-example.js`.

4. **[Compiler Explorer (245)](https://compiler.onnx9000.dev)**
   Interactive Web UI for pasting `.onnx` and seeing `.mlir` and `WGSL` updates in real-time.

5. **[Chrome Extension (246)](https://github.com/onnx9000/chrome-wvm-extension)**
   Service worker embedding executing `.wvm` models offline directly in the browser background.

6. **Cross Platform Parity (247)**
   The exact `.wvm` binary evaluates perfectly bit-for-bit on Node.js and Google Chrome.

7. **REST API Server (248)**
   Node.js/Express server loading `.wvm` models and exposing an OpenAI compatible `/v1/chat/completions` endpoint.

8. **WVM Model Gallery (249)**
   Pre-compiled `ResNet50.wvm`, `MobileNetV2.wvm`, `TinyLlama.wvm` are available on the HuggingFace Hub.

9. **Transition Docs (250)**
   See `docs/TUTORIAL_MLIR_LOWERING.md` for understanding the AOT vs JIT execution models.
