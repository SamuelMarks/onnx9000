# onnx9000: Usage Guide

This document provides a highly detailed, comprehensive guide on how to utilize the `onnx9000` framework. It covers everything from graph definition and JIT tracing in Python, to advanced compilation targets, native extension interop, and WebAssembly deployment.

**🔥 STATUS:** `onnx9000` is maturing rapidly. Over 200 standard ONNX operators have been implemented. The core IR parsing, autograd engine, static C++ memory planning, Apple Accelerate framework, WebAssembly SIMD backends, CUDA target, WebGPU, and advanced WebWorker RPC architectures are fully integrated and verified with 100% test and doc coverage across Python, C++, and TypeScript.


## Table of Contents

1. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
2. [The Python Frontend](#the-python-frontend)
    - [Tensors and Parameters](#tensors-and-parameters)
    - [The `@jit` Decorator](#the-jit-decorator)
    - [Supported Operations](#supported-operations)
3. [Graph Parsing & Exporting](#graph-parsing--exporting)
4. [The Compiler API (`onnx9000.compile`)](#the-compiler-api)
    - [Target: C++ (JIT Native)](#target-c-jit-native)
    - [Target: WebAssembly (WASM)](#target-webassembly-wasm)
5. [The Autograd Engine](#the-autograd-engine)
6. [Debugging and Profiling](#debugging-and-profiling)

---

## 1. Prerequisites & Environment Setup

`onnx9000` relies heavily on ahead-of-time (AOT) and just-in-time (JIT) compilation to turn graphs into executable code.

### Required Toolchains
- **Python:** Version 3.9 or higher.
- **C++ Compiler:** You must have a compiler supporting C++23. 
  - macOS: `clang++` (installed via Xcode command line tools, usually Apple Clang 14+ is sufficient for basic C++23 features used here like `std::expected` polyfills/native).
  - Linux: `g++` (GCC 12+) or `clang++` (Clang 15+).
- **Emscripten:** Required *only* if you are targeting WASM. You must have the `emsdk` installed, and `emcc` must be available in your system `$PATH`.
- **CMake:** Used under the hood by Pybind11 to structure the native extension compilation.

### Environment Variables
- `ONNX9000_CACHE_DIR`: By default, compiled C++ shared libraries are cached in `~/.cache/onnx9000/`. You can override this.
- `ONNX9000_DEBUG_CODEGEN`: Set to `1` to dump the raw generated `.cpp` file to standard output before compilation, which is invaluable for debugging compiler errors.
- `ONNX9000_USE_ACCELERATE`: Set to `1` (default on macOS) to link against the Apple Accelerate framework for fast matrix math (`vDSP`, `vForce`).
- `ONNX9000_USE_CUDA`: Set to `1` to emit CUDA device memory structures and utilize `CUDAExecutionProvider` primitives.

---

## 2. The Python Frontend

While `onnx9000` is perfectly capable of parsing `.onnx` files exported from PyTorch or TensorFlow, it also ships with a native, pure-Python tracing frontend.

### Tensors and Parameters

The fundamental objects in the `onnx9000` frontend are `Tensor` and `Parameter`. These are symbolic placeholders used to trace a computation graph. They do not hold concrete data arrays (like NumPy arrays) during tracing.

```python
from onnx9000 import Tensor, Parameter, DType

# A dynamic tensor. Only the shape structure and type are known.
# Use 'None' or strings for dynamic dimensions (e.g., batch size).
x = Tensor(shape=("batch_size", 3, 224, 224), dtype=DType.FLOAT32, name="input_image")

# A parameter (often representing model weights). 
# Parameters are typically considered static for inference, allowing aggressive constant folding in the compiler.
w = Parameter(shape=(64, 3, 7, 7), dtype=DType.FLOAT32, name="conv_weights")
```

### The `@jit` Decorator

The `@onnx9000.jit` decorator instruments a Python function so that when it is invoked with symbolic `Tensor` inputs, it traces the execution rather than running native Python math.

```python
import onnx9000

@onnx9000.jit
def my_residual_block(x, w1, w2):
    # Triggers onnx9000.ops.matmul
    hidden = x @ w1
    # Triggers onnx9000.ops.relu
    activated = onnx9000.ops.relu(hidden)
    # Residual connection
    out = (activated @ w2) + x
    return out
```

When you call `my_residual_block(x, w1, w2)`, the return value is not a computed tensor, but a `GraphBuilder` instance representing the parsed Abstract Syntax Tree (AST) of the traced operations.

### Supported Operations

Operations are exposed under `onnx9000.ops`. The frontend overrides standard Python magic methods (`__add__`, `__mul__`, `__matmul__`, etc.) to map seamlessly to ONNX primitives.

- **Math:** `add`, `sub`, `mul`, `div`, `matmul`, `pow`, `exp`, `log`, `abs`, `neg`.
- **Activations:** `relu`, `sigmoid`, `tanh`, `gelu`, `softmax`.
- **Reductions:** `reduce_sum`, `reduce_mean`, `reduce_max`, `reduce_min` (supports the `axes` argument).
- **Shape/Manipulation:** `reshape`, `transpose`, `concat`, `split`, `slice`, `gather`, `squeeze`, `unsqueeze`.
- **Neural Net Primitives:** `conv`, `batchnorm`, `maxpool`, `averagepool`.

Broadcasting rules strictly follow the standard ONNX/NumPy broadcasting semantics. 

---

## 3. Graph Parsing & Exporting

### Exporting Traced Graphs
Once you have traced a graph into a `GraphBuilder`, you can serialize it to a standard `.onnx` protobuf file. This file is 100% compliant with the ONNX specification and can be opened in tools like Netron.

```python
# Trace the graph
builder = my_residual_block(sym_x, sym_w1, sym_w2)

# Export to binary format
onnx9000.to_onnx(builder, "residual.onnx")

# Or get the raw protobuf string
proto_bytes = onnx9000.to_string(builder)
```

### Parsing Existing Models
To ingest an ONNX model trained in an external framework:

```python
from onnx9000 import parser

# Load graph into the internal IR
ir_graph = parser.parse_file("resnet50.onnx")

# Print the nodes in the IR
for node in ir_graph.nodes:
    print(f"{node.op_type}: {node.inputs} -> {node.outputs}")
```

---

## 4. The Compiler API

The crown jewel of `onnx9000` is its `compile` function. It takes an ONNX file or a `GraphBuilder` and transpiles it into executable code.

```python
onnx9000.compile(
    model: Union[str, GraphBuilder, ir.Graph],
    target: str = "cpp",
    out_dir: Optional[str] = None,
    optimize: bool = True,
    **kwargs
)
```

### Target: C++ (JIT Native)

When `target="cpp"` (the default), the compiler generates a `.cpp` file implementing your graph, wraps it with Pybind11 bindings, shells out to the C++ compiler to build a shared library, and dynamically imports it.

```python
model_executable = onnx9000.compile("model.onnx", target="cpp", optimize=True)

# model_executable is a callable Python object mapping to the native C++ inference engine.
outputs = model_executable(input_array_1, input_array_2)
```

**Memory Management in C++:**
The generated C++ code computes a single "Arena Size" required for all intermediate tensors. During instantiation, it allocates a single `std::vector<uint8_t>` or a raw heap array. During the `forward()` pass, intermediate tensors are simply `std::span` views into this arena. This guarantees zero fragmentation and lighting fast execution.

**Zero-Copy:** 
When passing NumPy arrays to the compiled model, Pybind11 will attempt zero-copy buffer protocol mapping. Ensure your NumPy arrays are contiguous (`np.ascontiguousarray`) and match the required `DType` to prevent implicit copies.

### Target: WebAssembly (WASM)

When `target="wasm"`, the workflow changes. `onnx9000` generates C++ code specifically annotated for Emscripten's `embind`, and invokes `emcc`.

```python
onnx9000.compile("model.onnx", target="wasm", out_dir="./dist")
```

This generates `model.js` and `model.wasm` in `./dist`.

**Browser Usage:**
```javascript
import Module from './dist/model.js';

Module().then(wasmModule => {
    // Instantiate the model class generated by onnx9000
    const model = new wasmModule.Model();
    
    // Create Float32Arrays for inputs
    const inputData = new Float32Array(1 * 3 * 224 * 224);
    // ... fill inputData ...
    
    // The WASM wrapper exposes specific set/get methods for tensors
    model.set_input("input_image", inputData);
    
    // Run the graph
    model.run();
    
    // Retrieve output
    const outputData = model.get_output("output_tensor");
    console.log(outputData);
    
    // Free C++ side memory
    model.delete(); 
});
```

---

## 5. The Autograd Engine

`onnx9000` contains a native Reverse-Mode Automatic Differentiation (AD) engine implemented in the `onnx9000.autograd` module. Unlike PyTorch, which builds a dynamic gradient tape at runtime, `onnx9000` performs **symbolic differentiation** on the IR before code generation.

This means you can trace a forward pass, invoke the autograd compiler, and generate a monolithic C++ kernel that computes both the forward pass and the exact gradients, all within a single statically planned memory arena.

```python
from onnx9000.autograd import vjp

# 1. Trace forward
builder = my_model(x, weights)

# 2. Compute Vector-Jacobian Product (Gradients)
# We want the gradients of the output with respect to 'weights'
backward_builder = vjp(builder, wrt=["weights"])

# 3. Compile the backward graph
backward_model = onnx9000.compile(backward_builder, target="cpp")

# Run forward + backward natively in C++
# outputs will contain the forward result AND the gradients for 'weights'
outputs, grad_weights = backward_model(x_data, weights_data)
```

This is profoundly powerful for on-device, edge training where shipping a full deep learning framework is impossible.

---

## 6. Debugging and Profiling

Given the transpilation architecture, debugging `onnx9000` involves inspecting the generated code.

1. **Viewing Generated Code:**
   Set the environment variable `ONNX9000_DEBUG_CODEGEN=1`. This will print the raw, pre-compiled C++ source directly to your terminal. It is highly recommended to inspect this to understand how your operations are lowered.

2. **Graph Visualization:**
   Always export your intermediate `GraphBuilder` objects to `.onnx` and open them in Netron to visually inspect the node connections and type inferences before attempting to compile them.

3. **Cache Clearing:**
   If you suspect the JIT cache is stale or corrupted, use the utility function:
   ```python
   from onnx9000 import clear_cache
   clear_cache()
   ```

4. **Debugging Types:**
   If the C++ compiler fails with template errors, it usually indicates a shape or type mismatch during the IR phase that slipped past the Python checks. Ensure your symbolic `Tensor` inputs have precisely correct shape tuples and `DType` enumerations.
