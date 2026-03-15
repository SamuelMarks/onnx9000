# ONNX3: Torch/TF Tracing Exporters Native Rewrite

## Introduction
**Target Projects:** `torch.onnx` (PyTorch) & `tf2onnx` (TensorFlow)
**New Home:** `src/onnx9000/frontends/`

The official `torch.onnx` and `tf2onnx` exporters are deeply embedded within massive, multi-gigabyte C++ frameworks. They rely on spinning up the full PyTorch/TensorFlow backend to trace graphs, execute JIT compiler passes, and serialize to the ONNX protobuf format. This makes it utterly impossible to compile these exporters for browser execution (via Pyodide) without downloading gigabytes of C++ bindings.

**The `onnx9000` Vision:** We are building a lightweight, pure-Python tracing frontend inspired by `TorchDynamo` and JAX's `jaxpr`. This `frontend` will use standard Python metaprogramming and operator overloading to trace arbitrary numerical code and generate `onnx9000` Intermediate Representation (IR) graphs *without* requiring the PyTorch or TensorFlow C++ binaries. A user can write a neural network in pure Python using `onnx9000.frontends.torch_like`, trace it instantly in the browser via Pyodide, and compile it to WebGPU/WASM.

## Exhaustive Implementation Checklist (300+ Items)

### Phase 1: Pure Python `Tensor` Frontend (Torch Parity)
- [x] **Step 001:** Implement the base `onnx9000.frontends.tensor.Tensor` class.
- [x] **Step 002:** Implement `__add__` overloading to map to ONNX `Add`.
- [x] **Step 003:** Implement `__sub__` overloading to map to ONNX `Sub`.
- [x] **Step 004:** Implement `__mul__` overloading to map to ONNX `Mul`.
- [x] **Step 005:** Implement `__truediv__` overloading to map to ONNX `Div`.
- [x] **Step 006:** Implement `__pow__` overloading to map to ONNX `Pow`.
- [x] **Step 007:** Implement `__mod__` overloading to map to ONNX `Mod`.
- [x] **Step 008:** Implement `__matmul__` overloading to map to ONNX `MatMul`.
- [x] **Step 009:** Implement `__neg__` overloading to map to ONNX `Neg`.
- [x] **Step 010:** Implement `__abs__` overloading to map to ONNX `Abs`.
- [x] **Step 011:** Implement `__getitem__` overloading to map to ONNX `Slice` / `Gather`.
- [x] **Step 012:** Implement `__setitem__` overloading to map to ONNX `ScatterND` (immutable update).
- [x] **Step 013:** Implement `sum()` method to map to ONNX `ReduceSum`.
- [x] **Step 014:** Implement `mean()` method to map to ONNX `ReduceMean`.
- [x] **Step 015:** Implement `max()` method to map to ONNX `ReduceMax`.
- [x] **Step 016:** Implement `min()` method to map to ONNX `ReduceMin`.
- [x] **Step 017:** Implement `transpose()` and `.T` to map to ONNX `Transpose`.
- [x] **Step 018:** Implement `reshape()` and `view()` to map to ONNX `Reshape`.
- [x] **Step 019:** Implement `squeeze()` and `unsqueeze()` to map to ONNX `Squeeze`/`Unsqueeze`.
- [x] **Step 020:** Implement `flatten()` to map to ONNX `Flatten`.
- [x] **Step 021:** Implement `expand()` and `broadcast_to()` to map to ONNX `Expand`.
- [x] **Step 022:** Implement `contiguous()` as a logical no-op (or explicit copy in WebGPU).
- [x] **Step 023:** Implement `type()` and `to()` casting to map to ONNX `Cast`.
- [x] **Step 024:** Implement elementwise math methods: `exp()`, `log()`, `sqrt()`, `sin()`, `cos()`, `tan()`.
- [x] **Step 025:** Implement elementwise math methods: `asin()`, `acos()`, `atan()`, `sinh()`, `cosh()`.
- [x] **Step 026:** Implement activation methods: `relu()`, `sigmoid()`, `tanh()`, `gelu()`, `softmax()`.
- [x] **Step 027:** Implement comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`) mapping to ONNX logic ops.
- [x] **Step 028:** Implement boolean ops (`__and__`, `__or__`, `__xor__`, `~`) mapping to ONNX bitwise ops.
- [x] **Step 029:** Implement `where()` mapping to ONNX `Where`.
- [x] **Step 030:** Implement `clip()` and `clamp()` mapping to ONNX `Clip`.
- [x] **Step 031:** Implement `argmax()` and `argmin()` mapping to ONNX `ArgMax`/`ArgMin`.
- [x] **Step 032:** Implement `gather()` and `scatter()` mapping to ONNX equivalents.
- [x] **Step 033:** Implement `masked_select()` mapping to boolean indexing.
- [x] **Step 034:** Implement `nonzero()` mapping to ONNX `NonZero`.
- [x] **Step 035:** Implement scalar and list conversions (`item()`, `tolist()`).
- [x] **Step 036:** Implement `numpy()` conversion method.
- [x] **Step 037:** Support creating tensors from Python lists/scalars.
- [x] **Step 038:** Support creating tensors from NumPy arrays.
- [x] **Step 039:** Support creating tensors from pure memory buffers (for Pyodide integration).
- [x] **Step 040:** Implement shape broadcasting rules exactly mirroring PyTorch during operator overloading.
- [x] **Step 041:** Write 100+ unit tests guaranteeing exact PyTorch mathematical parity for all methods.
- [x] **Step 042:** Implement `requires_grad` property tracking for the Autograd engine.
- [x] **Step 043:** Implement `.grad` property accumulation.
- [x] **Step 044:** Implement `.detach()` method to stop gradient tracking.
- [x] **Step 045:** Implement `.clone()` method.
- [x] **Step 046:** Ensure `Tensor` instances immutability (functional graph generation).
- [x] **Step 047:** Finalize Phase 1 pure Python Torch-like API.

### Phase 2: Pure Python `nn.Module` Framework
- [x] **Step 048:** Implement the base `onnx9000.frontends.nn.Module` class.
- [x] **Step 049:** Implement `Parameter` subclass of `Tensor`.
- [x] **Step 050:** Implement `register_parameter()` and `register_buffer()` logic.
- [x] **Step 051:** Implement hierarchical `add_module()` tracking.
- [x] **Step 052:** Implement `parameters()`, `named_parameters()`, `buffers()`, `children()`, `modules()` generators.
- [x] **Step 053:** Implement `state_dict()` recursive extraction mapping to ONNX names.
- [x] **Step 054:** Implement `load_state_dict()` logic.
- [x] **Step 055:** Implement `to()` device casting for all module parameters.
- [x] **Step 056:** Implement `train()` and `eval()` mode switching (affecting Dropout/BatchNorm).
- [x] **Step 057:** Implement `apply()` method for recursive initialization.
- [x] **Step 058:** Implement `nn.Linear` (wrapping `MatMul` + `Add`).
- [x] **Step 059:** Implement `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d` (wrapping ONNX `Conv`).
- [x] **Step 060:** Implement `nn.ConvTranspose1d`, `nn.ConvTranspose2d`.
- [x] **Step 061:** Implement `nn.MaxPool1d`, `nn.MaxPool2d`, `nn.AvgPool1d`, `nn.AvgPool2d`.
- [x] **Step 062:** Implement `nn.AdaptiveAvgPool2d` (mapping to `GlobalAveragePool`).
- [x] **Step 063:** Implement `nn.BatchNorm1d`, `nn.BatchNorm2d`, `nn.BatchNorm3d`.
- [x] **Step 064:** Implement `nn.LayerNorm` and `nn.GroupNorm`.
- [x] **Step 065:** Implement `nn.InstanceNorm1d`, `nn.InstanceNorm2d`.
- [x] **Step 066:** Implement `nn.Dropout`, `nn.Dropout2d` (conditional execution based on `training` flag).
- [x] **Step 067:** Implement `nn.Embedding` (wrapping `Gather`).
- [x] **Step 068:** Implement `nn.RNN`, `nn.LSTM`, `nn.GRU` (wrapping ONNX complex loop ops).
- [x] **Step 069:** Implement `nn.Sequential` and `nn.ModuleList` containers.
- [x] **Step 070:** Implement `nn.ModuleDict` and `nn.ParameterList`.
- [x] **Step 071:** Implement `nn.Identity`.
- [x] **Step 072:** Implement `nn.Flatten` and `nn.Unflatten`.
- [x] **Step 073:** Implement custom weight initialization utilities (`nn.init.kaiming_normal_`, `xavier_uniform_`).
- [x] **Step 074:** Implement `nn.functional` matching Torch API (e.g., `F.relu`, `F.conv2d`, `F.pad`).
- [x] **Step 075:** Implement `F.interpolate` (mapping to ONNX `Resize`).
- [x] **Step 076:** Implement `F.one_hot` (mapping to ONNX `OneHot`).
- [x] **Step 077:** Write rigorous tests asserting `state_dict` keys perfectly match PyTorch keys.
- [x] **Step 078:** Write tests verifying output parity between pure PyTorch `nn.Conv2d` and our `nn.Conv2d`.
- [x] **Step 079:** Support arbitrary nesting of `Module` definitions.
- [x] **Step 080:** Implement `zero_grad()` method recursively.
- [x] **Step 081:** Finalize Phase 2 `nn.Module` Architecture.

### Phase 3: Symbolic Execution & Graph Tracing (JIT)
- [x] **Step 082:** Implement the `@jit` decorator in `src/onnx9000/frontends/jit/`.
- [x] **Step 083:** Implement a `Tracer` context manager that intercepts all `Tensor` operations.
- [x] **Step 084:** Implement a `Proxy` tensor object that records operations instead of executing them.
- [x] **Step 085:** When a Python function is called, convert inputs to `Proxy` tensors.
- [x] **Step 086:** Map recorded `Proxy` operations directly to `onnx9000.core.ir.Node` instances.
- [x] **Step 087:** Generate a deterministic naming scheme for traced intermediate tensors.
- [x] **Step 088:** Handle constant folding during tracing (e.g., `x + 2` creates a `Constant` node).
- [x] **Step 089:** Handle standard Python control flow (`if`, `for`, `while`).
- [x] **Step 090:** Implement graph flattening for standard straight-line code.
- [x] **Step 091:** Detect and raise errors for data-dependent control flow (dynamic `if x > 0:`).
- [x] **Step 092:** Implement static loop unrolling for predictable `for` loops during tracing.
- [x] **Step 093:** Implement Python abstract syntax tree (AST) parsing (`ast.parse`) for advanced control flow extraction.
- [x] **Step 094:** Translate Python `if/else` AST blocks into ONNX `If` subgraphs.
- [x] **Step 095:** Translate Python `for/while` AST blocks into ONNX `Loop` subgraphs.
- [x] **Step 096:** Capture closed-over variables (closures) correctly during AST translation.
- [x] **Step 097:** Implement shape inference during tracing to validate graph correctness immediately.
- [x] **Step 098:** Handle multiple outputs from a traced function.
- [x] **Step 099:** Handle dictionary and tuple outputs from a traced function.
- [x] **Step 100:** Implement tracing for `nn.Module.forward()` calls.
- [x] **Step 101:** Automatically extract `nn.Parameter` tensors as explicit ONNX graph `inputs` during `Module` tracing.
- [x] **Step 102:** Implement caching of traced graphs (hashing based on input shapes/dtypes).
- [x] **Step 103:** Provide a mechanism to trace dynamic axes (e.g., variable batch size `N`).
- [x] **Step 104:** Implement `torch.jit.trace` equivalent API (`onnx9000.frontends.jit.trace()`).
- [x] **Step 105:** Implement `torch.jit.script` equivalent API (`onnx9000.frontends.jit.script()`).
- [x] **Step 106:** Provide fallback to eager execution if tracing fails.
- [x] **Step 107:** Implement comprehensive tracing diagnostics (pointing to the exact Python line that failed).
- [x] **Step 108:** Ensure standard Python functions (`len`, `print`) are ignored or translated safely.
- [x] **Step 109:** Trace NumPy functions applied to our `Tensor` objects seamlessly.
- [x] **Step 110:** Ensure the generated `ir.Graph` exactly matches standard ONNX Opset 18+ spec.
- [x] **Step 111:** Finalize Phase 3 JIT Tracer.

### Phase 4: Exporter Serialization & Pyodide Hooks
- [x] **Step 112:** Implement `onnx9000.export()` API matching `torch.onnx.export` signature.
- [x] **Step 113:** Serialize the traced `ir.Graph` directly to `onnx_pb2.ModelProto`.
- [x] **Step 114:** Support `opset_version` targets.
- [x] **Step 115:** Support `do_constant_folding` flag during export.
- [x] **Step 116:** Support `input_names` and `output_names` overrides.
- [x] **Step 117:** Support `dynamic_axes` definitions exactly matching PyTorch API.
- [x] **Step 118:** Handle saving large models (>2GB) using ONNX external data format natively.
- [x] **Step 119:** Ensure `export()` runs successfully in Pyodide without any external C++ dependencies.
- [x] **Step 120:** Build a Pyodide-specific wheel for the frontend package.
- [x] **Step 121:** Implement a JS wrapper to invoke the Python exporter from the browser.
- [x] **Step 122:** Write tests validating models exported in-browser against standard ONNX Runtime.
- [x] **Step 123:** Implement memory optimizations during export to prevent Pyodide OOM.
- [x] **Step 124:** Handle serializing quantized weights (INT8/FP16) natively.
- [x] **Step 125:** Implement an ONNX format visualizer to verify exported topology.
- [x] **Step 126:** Implement `keep_initializers_as_inputs` flag logic.
- [x] **Step 127:** Implement `custom_opsets` dictionary registration.
- [x] **Step 128:** Map standard Torchvision/HuggingFace model architectures to the frontend.
- [x] **Step 129:** Export an exact replica of ResNet-18 using pure Python.
- [x] **Step 130:** Export an exact replica of MobileNetV2 using pure Python.
- [x] **Step 131:** Export an exact replica of GPT-2 using pure Python.
- [x] **Step 132:** Finalize Phase 4 Serialization.

### Phase 5: TensorFlow / JAX Translation Layers
- [x] **Step 133:** Create the `onnx9000.frontends.tf` namespace.
- [x] **Step 134:** Create the `onnx9000.frontends.jax` namespace.
- [x] **Step 135:** Implement an importer that parses TensorFlow `SavedModel` protobufs in pure Python.
- [x] **Step 136:** Implement a mapping layer from TF `GraphDef` operations to ONNX9000 `ir.Node`.
- [x] **Step 137:** Handle TensorFlow NHWC to ONNX NCHW data layout transformations automatically.
- [x] **Step 138:** Implement a mapping layer from JAX `jaxpr` to ONNX9000 `ir.Node`.
- [x] **Step 139:** Handle JAX `vmap` (vectorization) translation to ONNX broadcasting ops.
- [x] **Step 140:** Handle JAX `pmap` translation to ONNX partitioning/split ops.
- [x] **Step 141:** Handle JAX `grad` translation using our Autograd engine.
- [x] **Step 142:** Write parity tests for TF Core operators (Conv2D, MatMul, Relu).
- [x] **Step 143:** Write parity tests for JAX Core operations.
- [x] **Step 144:** Ensure all importers run natively in Pyodide.
- [x] **Step 145:** Provide a unified `onnx9000.load('model.pb')` interface.
- [x] **Step 146:** Implement fallback mechanisms for unsupported TF/JAX ops.
- [x] **Step 147:** Finalize Phase 5 Translation Layers.

### Phase 6: Final API Polish & Compatibility
- [x] **Step 148:** Ensure `import torch` can be aliased to `import onnx9000.frontends.torch_like as torch` seamlessly.
- [x] **Step 149:** Test drop-in replacement on a standard PyTorch tutorial script.
- [x] **Step 150:** Document the architectural differences and tracing limitations compared to native PyTorch.
- [x] **Step 151:** Publish `@onnx9000/frontends` documentation.
- [x] **Step 152:** Finalize ONNX3 Tracing Exporters.

