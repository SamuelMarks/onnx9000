# Usage Guide

`onnx9000` is a comprehensive polyglot ecosystem comprising Python and TypeScript/JavaScript packages. The following guide covers installation, the Python Execution Pipeline, the TypeScript Web Backends, and the Unified CLI tool.

## Installation

### Python (via pip or uv)

```bash
# In your Python 3.9+ environment
pip install onnx9000-core onnx9000-backend-native onnx9000-optimizer
```

### TypeScript / JavaScript (via pnpm or npm)

```bash
# For Node.js, Deno, Bun, or Browser (Webpack/Vite) targets
pnpm add @onnx9000/core @onnx9000/backend-web
```

## 🐍 Python Examples

### 1. Zero-Dependency Model Loading & Inspection

The core of `onnx9000` does not rely on the massive `onnx` protobuf Python package or `numpy`. It uses raw Python data structures for memory efficiency.

```python
from onnx9000.core.parser.core import load

# Parses the raw protobuf structure into an in-memory `Graph` AST instantly.
graph = load("mobilenetv2.onnx")

print(f"Model Name: {graph.name}")
print(f"Inputs: {graph.inputs}")
print(f"Number of Tensors: {len(graph.tensors)}")

# Optional: Run static shape inference on dynamic graphs
from onnx9000.core.shape_inference import infer_shapes_and_types
infer_shapes_and_types(graph)
```

### 2. Execution (Native CPU)

`onnx9000-backend-native` exposes `ExecutionProviders` that allocate contiguous memory arenas and evaluate nodes using `ctypes` dispatches to native math libraries (e.g., OpenBLAS or Apple Accelerate).

```python
import numpy as np
from onnx9000.core.parser.core import load
from onnx9000.core.ir import Tensor
from onnx9000.core.dtypes import DType
from onnx9000.backends.session import InferenceSession, SessionOptions
from onnx9000.backends.cpu.executor import CPUExecutionProvider

graph = load("model.onnx")

# Set up Session Configuration
options = SessionOptions(execution_mode="SEQUENTIAL", enable_profiling=True)

# Orchestrate execution via the CPU Provider
session = InferenceSession(graph, providers=[CPUExecutionProvider({})], options=options)

# Generate Mock Data
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Wrap NumPy array in the internal `onnx9000.Tensor` (Zero-copy DLPack mapping)
input_tensor = Tensor(name="input_1", shape=(1, 3, 224, 224), dtype=DType.FLOAT32, data=input_data.tobytes())

# Execute
outputs = session.run(output_names=["output_1"], input_feed={"input_1": input_tensor})

# Access resulting memoryview
print(outputs["output_1"].data)
```

## 🌐 TypeScript Examples

### 1. WebGPU / WebNN Execution

The `@onnx9000/backend-web` package translates the `Graph` AST into dynamic WebGPU compute shaders or `navigator.ml` WebNN builder contexts natively within the browser, enabling near-native FPS for large vision and language models.

```typescript
import { load } from '@onnx9000/core';
import { InferenceSession, WebGPUProvider } from '@onnx9000/backend-web';

async function runModel(modelUrl: string) {
  // Fetch and parse the ONNX model into the TS AST
  const response = await fetch(modelUrl);
  const buffer = await response.arrayBuffer();
  const graph = load(buffer);

  // Initialize the WebGPU execution provider
  const webgpuProvider = new WebGPUProvider();
  await webgpuProvider.initialize();

  // Create session
  const session = new InferenceSession(graph, [webgpuProvider]);

  // Create input tensor natively
  const inputData = new Float32Array(1 * 3 * 224 * 224).fill(0.5);
  const inputFeed = {
    input: { data: inputData, shape: [1, 3, 224, 224], dtype: 'float32' },
  };

  // Run inference asynchronously via WebGPU shaders
  const results = await session.run(['output'], inputFeed);

  console.log('Inference complete!', results['output'].data);
}
```

## 💻 Unified CLI (`apps/cli`)

The `onnx9000` CLI acts as a comprehensive entry point to the entire Python optimization and compilation toolchain.

```bash
# Inspect the topology, memory usage, and FLOP count of an ONNX file
onnx9000 inspect ./model.onnx

# Apply Level 3 (Transformer fusions, Gelu, RoPE) optimizations
onnx9000 optimize ./model.onnx --level 3 --output ./optimized.onnx

# Quantize a model to Int8 natively
onnx9000 quantize ./model.onnx --format int8 --output ./quantized.onnx

# Convert a legacy Keras H5 file directly to ONNX (Zero TensorFlow dependency required)
onnx9000 convert --src keras --dst onnx ./model.h5

# Launch the local Netron-style Web Visualizer Server
onnx9000 serve ./model.onnx
```
