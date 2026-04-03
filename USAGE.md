# Usage Guide

`onnx9000` is a comprehensive polyglot ecosystem. By entirely replacing C++ tooling, we offer a truly cross-platform, zero-dependency environment for modern Machine Learning.

## Installation

### Python (via uv or pip)

```bash
# Minimal core for parsing/editing
uv pip install onnx9000-core

# Hardware execution, optimizers, and converters
uv pip install onnx9000-backend-native onnx9000-optimizer onnx9000-converters

# Everything
uv pip install "onnx9000[all]"
```

### TypeScript / JavaScript (via pnpm)

```bash
# Core AST and parsers
pnpm add @onnx9000/core

# Web backends (WebGPU / WebNN / WASM)
pnpm add @onnx9000/backend-web

# Higher-level pipelines (LLMs, SD)
pnpm add @onnx9000/transformers @onnx9000/diffusers
```

---

## 🐍 Python Ecosystem

### Zero-Dependency Parsers

`onnx9000-core` reads `.onnx`, `.pb`, `.h5`, `.tflite`, `.gguf`, and `.safetensors` using raw Python data structures.

```python
from onnx9000.core.parser.core import load
from onnx9000.core.shape_inference import infer_shapes_and_types

# Parses the structure into an in-memory Graph AST
graph = load("mobilenetv2.onnx")

# Run strict static shape inference
infer_shapes_and_types(graph)

# Extract a subgraph surgically
from onnx9000.core.graph_surgeon import extract_subgraph
sub_graph = extract_subgraph(graph, input_names=["conv1_out"], output_names=["layer3_out"])
```

### Hardware-Native Execution

`onnx9000-backend-native` routes operations via `ctypes` to native math libraries.

```python
import numpy as np
from onnx9000.core.parser.core import load
from onnx9000.backends.session import InferenceSession
from onnx9000.backends.cpu.executor import CPUExecutionProvider
from onnx9000.backends.cuda.executor import CUDAExecutionProvider

graph = load("model.onnx")

# Orchestrate execution via CPU or CUDA Providers
session = InferenceSession(
    graph,
    providers=[CUDAExecutionProvider(), CPUExecutionProvider()]
)

# Run inference
input_data = {"input_1": np.random.randn(1, 3, 224, 224).astype(np.float32)}
outputs = session.run(output_names=["output_1"], input_feed=input_data)
```

### Optimization & Quantization

`onnx9000-optimizer` offers algebraic simplification and INT4/INT8 quantization.

```python
from onnx9000.optimizer import optimize, quantize, QuantizationConfig

# Apply Level 3 fusions (GELU, RoPE, LayerNorm)
optimized_graph = optimize(graph, level=3)

# INT8 Quantization
q_config = QuantizationConfig(weight_type="int8", activation_type="int8")
quantized_graph = quantize(optimized_graph, q_config)
```

---

## 🌐 TypeScript & Web Ecosystem

### WebGPU / WebNN Inference

`@onnx9000/backend-web` turns your AST into optimized WebGPU WGSL shaders or leverages WebNN for NPU access.

```typescript
import { load } from '@onnx9000/core';
import { InferenceSession, WebGPUProvider, WebNNProvider } from '@onnx9000/backend-web';

async function runModel(modelUrl: string) {
  const buffer = await (await fetch(modelUrl)).arrayBuffer();
  const graph = load(buffer);

  const provider = new WebNNProvider(); // or WebGPUProvider
  await provider.initialize();

  const session = new InferenceSession(graph, [provider]);

  const inputData = new Float32Array(1 * 3 * 224 * 224).fill(0.5);
  const results = await session.run(['output'], {
    input: { data: inputData, shape: [1, 3, 224, 224], dtype: 'float32' },
  });
}
```

### High-level GenAI (Diffusers & Transformers)

You can run full pipelines using the built-in `@onnx9000/diffusers` or `@onnx9000/transformers`.

```typescript
import { StableDiffusionPipeline } from '@onnx9000/diffusers';

const pipe = await StableDiffusionPipeline.fromPretrained('runwayml/stable-diffusion-v1-5');
const image = await pipe.generate('a futuristic city skyline at sunset');
```

### TensorFlow.js Drop-in Shim

Migrate `@tensorflow/tfjs` projects with zero code changes.

```typescript
import * as tf from '@onnx9000/tfjs-shim';

const model = await tf.loadGraphModel('model.json');
const tensor = tf.tensor([1, 2, 3, 4], [2, 2]);
const output = model.predict(tensor);
output.print();
```

---

## 💻 Unified CLI (`onnx9000`)

```bash
# Inspect a model
onnx9000 inspect ./model.onnx

# Optimize and Quantize
onnx9000 optimize ./model.onnx --level 3 --output ./optimized.onnx
onnx9000 quantize ./model.onnx --format int8 --output ./quantized.onnx

# Convert formats (No original frameworks required)
onnx9000 convert --src keras --dst onnx ./model.h5
onnx9000 convert --src onnx --dst cpp ./model.onnx --output model.h
onnx9000 convert --src onnx --dst gguf ./model.onnx --output model.gguf

# Launch Netron UI
onnx9000 ui ./model.onnx
```