# Usage Guide

`onnx9000` is a comprehensive polyglot ecosystem comprising Python and TypeScript/JavaScript packages. By entirely replacing C++ tooling, we offer a truly cross-platform, zero-dependency environment for modern Machine Learning.

This guide covers installation, Python APIs, TypeScript Web APIs, Edge Serving, Generative AI, and the Unified CLI tool.

## Table of Contents

1. [Installation](#installation)
2. [🐍 Python Ecosystem](#python-ecosystem)
   - [Zero-Dependency Parsers & Inspection](#zero-dependency-parsers-inspection)
   - [Hardware-Native Execution](#hardware-native-execution)
   - [Model Optimization & Quantization](#model-optimization-quantization)
   - [Autograd & On-Device Training](#autograd-on-device-training)
   - [Framework Converters](#framework-converters)
3. [🌐 TypeScript & Web Ecosystem](#typescript-web-ecosystem)
   - [WebGPU & WebNN Inference](#webgpu-webnn-inference)
   - [TensorFlow.js Drop-in Shim](#tensorflow-js-drop-in-shim)
   - [Serverless Edge Serving (Cloudflare/Bun)](#serverless-edge-serving)
   - [AOT Compilation to WASM/C++](#aot-compilation-to-wasm-c)
   - [Generative AI & Diffusers](#generative-ai-diffusers)
4. [💻 Unified CLI (`onnx9000`)](#unified-cli-onnx9000)

---

## Installation

### Python (via pip or uv)

You can install individual packages or the entire toolkit. No C++ compiler or `cmake` is required.

```bash
# Minimal core for parsing/editing
uv pip install onnx9000-core

# Hardware execution, optimizers, and converters
uv pip install onnx9000-backend-native onnx9000-optimizer onnx9000-converters

# Everything
uv pip install "onnx9000[all]"
```

### TypeScript / JavaScript (via pnpm or npm)

Packages are modular and tree-shakeable for browser, Node.js, Bun, and Deno.

```bash
# Core AST and parsers
pnpm add @onnx9000/core

# Web backends (WebGPU / WebNN / WASM)
pnpm add @onnx9000/backend-web

# Higher-level pipelines (LLMs, SD)
pnpm add @onnx9000/transformers @onnx9000/diffusers
```

---

(python-ecosystem)=

## 🐍 Python Ecosystem

(zero-dependency-parsers-inspection)=

### Zero-Dependency Parsers & Inspection

The core `onnx9000-core` package does not rely on the massive `onnx` protobuf package. It reads `.onnx`, `.pb`, `.h5`, `.tflite`, `.gguf`, and `.safetensors` using raw Python data structures for memory efficiency and instant loading.

```python
from onnx9000.core.parser.core import load
from onnx9000.core.shape_inference import infer_shapes_and_types

# Parses the raw protobuf structure into an in-memory `Graph` AST instantly
graph = load("mobilenetv2.onnx")

print(f"Model: {graph.name} | IR Version: {graph.ir_version}")
print(f"Inputs: {[i.name for i in graph.inputs]}")
print(f"Tensors: {len(graph.tensors)}")

# Run strict static shape inference
infer_shapes_and_types(graph)

# Extract a subgraph surgically
from onnx9000.core.graph_surgeon import extract_subgraph
sub_graph = extract_subgraph(graph, input_names=["conv1_out"], output_names=["layer3_out"])
```

(hardware-native-execution)=

### Hardware-Native Execution

`onnx9000-backend-native` dynamically routes operations via `ctypes` to native math libraries (Apple Accelerate, CUDA, TensorRT) without the overhead of massive C++ bindings.

```python
import numpy as np
from onnx9000.core.parser.core import load
from onnx9000.core.ir import Tensor
from onnx9000.core.dtypes import DType
from onnx9000.backends.session import InferenceSession, SessionOptions
from onnx9000.backends.cpu.executor import CPUExecutionProvider
from onnx9000.backends.cuda.executor import CUDAExecutionProvider

graph = load("model.onnx")

options = SessionOptions(execution_mode="SEQUENTIAL", enable_profiling=True)

# Orchestrate execution via CPU or CUDA Providers
session = InferenceSession(
    graph,
    providers=[CUDAExecutionProvider(), CPUExecutionProvider()],
    options=options
)

# Zero-copy DLPack mapping with internal Tensor representation
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_tensor = Tensor(
    name="input_1",
    shape=(1, 3, 224, 224),
    dtype=DType.FLOAT32,
    data=input_data.tobytes()
)

outputs = session.run(output_names=["output_1"], input_feed={"input_1": input_tensor})
print("Result shape:", outputs["output_1"].shape)
```

(model-optimization-quantization)=

### Model Optimization & Quantization

The `onnx9000-optimizer` module offers state-of-the-art algebraic simplification (constant folding, fusion) and advanced INT4/INT8 quantization, fully replacing `onnx-simplifier` and `optimum`.

```python
from onnx9000.optimizer import optimize, quantize, QuantizationConfig

# Apply Level 3 fusions (GELU, RoPE, LayerNorm) and constant folding
optimized_graph = optimize(graph, level=3)

# INT8/INT4 Quantization
q_config = QuantizationConfig(
    weight_type="int8",
    activation_type="int8",
    per_channel=True,
    symmetric=True
)
quantized_graph = quantize(optimized_graph, q_config)
```

(autograd-on-device-training)=

### Autograd & On-Device Training

`onnx9000-toolkit` handles Ahead-of-Time (AOT) symbolic autograd. It takes a forward-pass ONNX graph and compiles the backward pass (VJPs) directly into the graph.

```python
from onnx9000.toolkit.autograd import add_backward_pass

# Given an inference graph with a defined loss node
training_graph = add_backward_pass(graph, loss_node="cross_entropy_loss")

# The resulting graph now accepts gradients and computes weight updates
print("Trainable parameters:", training_graph.get_trainable_initializers())
```

(framework-converters)=

### Framework Converters

Convert legacy and modern frameworks seamlessly into ONNX natively, or convert ONNX to other formats like TFLite and GGUF using bi-directional transpilation.

```python
from onnx9000.converters.tf import tf2onnx
from onnx9000.converters.gguf import onnx2gguf

# Convert a TensorFlow SavedModel to ONNX without TensorFlow installed!
onnx_graph = tf2onnx("/path/to/saved_model")

# Transpile an ONNX model natively to GGUF format for llama.cpp
gguf_bytes = onnx2gguf(onnx_graph, use_f16=True)
with open("model.gguf", "wb") as f:
    f.write(gguf_bytes)
```

---

(typescript-web-ecosystem)=

## 🌐 TypeScript & Web Ecosystem

(webgpu-webnn-inference)=

### WebGPU & WebNN Inference

The `@onnx9000/backend-web` package turns your TS AST into highly optimized WebGPU WGSL shaders or `navigator.ml` WebNN contexts natively in the browser.

```typescript
import { load } from '@onnx9000/core';
import { InferenceSession, WebGPUProvider, WebNNProvider } from '@onnx9000/backend-web';

async function runVisionModel(modelUrl: string) {
  const buffer = await (await fetch(modelUrl)).arrayBuffer();
  const graph = load(buffer);

  // Initialize providers: Try WebNN first, fallback to WebGPU
  const webnn = new WebNNProvider({ deviceType: 'npu' });
  const webgpu = new WebGPUProvider({ powerPreference: 'high-performance' });

  await webnn.initialize().catch(() => webgpu.initialize());

  const session = new InferenceSession(graph, [webnn, webgpu]);

  const inputData = new Float32Array(1 * 3 * 224 * 224).fill(0.5);
  const results = await session.run(['output'], {
    input: { data: inputData, shape: [1, 3, 224, 224], dtype: 'float32' },
  });

  console.log('Inference complete!', results['output'].data);
}
```

(tensorflow-js-drop-in-shim)=

### TensorFlow.js Drop-in Shim

Migrate old `@tensorflow/tfjs` projects to `onnx9000` with zero code changes using our shim.

```typescript
// Replace: import * as tf from '@tensorflow/tfjs';
import * as tf from '@onnx9000/tfjs-shim';

// Uses ONNX/WebGPU under the hood, but exposes the TFJS API
const model = await tf.loadGraphModel('model.json');
const tensor = tf.tensor([1, 2, 3, 4], [2, 2]);
const output = model.predict(tensor);
output.print();
```

(serverless-edge-serving)=

### Serverless Edge Serving

The `@onnx9000/serve` module provides an event-loop driven, dynamic batching inference server. It's perfectly designed for Cloudflare Workers, Bun, and Deno.

```typescript
// worker.ts (Cloudflare Worker or Bun)
import { serve } from '@onnx9000/serve';
import { WebAssemblyProvider } from '@onnx9000/backend-web';

const app = serve({
  model: 's3://models/bert.onnx',
  provider: new WebAssemblyProvider({ simd: true, threads: 4 }),
  batching: { maxBatchSize: 8, timeoutMs: 10 },
});

export default app;
```

(aot-compilation-to-wasm-c)=

### AOT Compilation to WASM / C++

Compile an ONNX model strictly to a standalone executable or WebAssembly module. This entirely skips the "interpreter" overhead at runtime.

```typescript
import { compile } from '@onnx9000/compiler';

// Compiles the graph into a standalone C++23 header
const cppCode = await compile(graph, { target: 'cpp23', staticMemory: true });

// Compiles the graph into a raw WebAssembly bytecode array
const wasmModule = await compile(graph, { target: 'wasm', useSIMD: true });
```

(generative-ai-diffusers)=

### Generative AI & Diffusers

Direct integration for Hugging Face pipelines natively in WebWorkers using customized W4A16 packed weights.

```typescript
import { pipeline } from '@onnx9000/transformers';

// Automatic text-generation using an optimized ONNX-based LLM in WebGPU
const generate = await pipeline('text-generation', 'onnx-community/Llama-3-8B-Instruct', {
  device: 'webgpu',
  dtype: 'q4f16',
});

const response = await generate('What is the future of ML?', { max_new_tokens: 100 });
console.log(response);
```

---

(unified-cli-onnx9000)=

## 💻 Unified CLI (`onnx9000`)

The `onnx9000` CLI is a "Swiss Army Knife" for the entire ML pipeline. It brings all the capabilities of the Python and JS packages into a single terminal command.

```bash
# 🔍 Inspection
# Inspect the topology, memory usage, and FLOP count of an ONNX file
onnx9000 inspect ./model.onnx

# ⚡ Optimization & Quantization
# Apply Level 3 (Transformer fusions, Gelu, RoPE) optimizations
onnx9000 optimize ./model.onnx --level 3 --output ./optimized.onnx

# Quantize a model to Int8 natively
onnx9000 quantize ./model.onnx --format int8 --output ./quantized.onnx

# 🔄 Converters
# Convert a legacy Keras H5 file directly to ONNX (Zero TensorFlow dependency required)
onnx9000 convert --src keras --dst onnx ./model.h5

# Convert an ONNX model to standalone C++ code
onnx9000 convert --src onnx --dst cpp ./model.onnx --output model.h

# Convert an ONNX model to an MLIR Graph
onnx9000 convert --src onnx --dst mlir ./model.onnx --output graph.mlir
onnx9000 convert --src onnx --dst gguf ./model.onnx --output model.gguf

# 🚀 Serverless & UI
# Launch the local Netron-style WebGL Visualizer Server
onnx9000 ui ./model.onnx

# Launch a dynamic batching inference API on port 8080
onnx9000 serve ./model.onnx --port 8080 --batch-size 16
```
