# ONNX5: ONNX-Simplifier Native Rewrite

## Introduction
**Target Project:** [ONNX-Simplifier](https://github.com/daquexian/onnx-simplifier)
**New Home:** `src/onnx9000/optimize/simplifier/`

The community-favorite `onnx-simplifier` tool drastically cleans up messy ONNX graphs exported from PyTorch/TensorFlow (removing redundant casts, identities, and pre-computing constant math). However, it relies heavily on spinning up the full ONNX Runtime C++ backend to execute the constant folding passes and evaluate subgraphs. This breaks our pure-Python/Pyodide execution constraint.

**The `onnx9000` Vision:** We are implementing a rigorous pure-Python algebraic optimizer, dead-code eliminator, and constant folding engine. By natively evaluating ONNX operators within Python using NumPy (or our own `Tensor` math backend), we can perform aggressive graph simplifications completely AOT (Ahead-of-Time). This drastically shrinks exported graphs before they hit restrictive browser execution contexts, ensuring fast load times and minimal WASM memory usage, all without requiring a C++ runtime to optimize the model.

## Exhaustive Implementation Checklist (300+ Items)

### Phase 1: Pure Python Constant Folding Engine
- [x][x][x] **Step 001:** Create the base `ConstantFoldingPass(GraphPass)` in `src/onnx9000/optimize/simplifier/passes/constant_folding.py`.
- [x][x][x] **Step 002:** Implement a topological traversal to identify nodes where all inputs are `Constant` or `initializer`.
- [x][x][x] **Step 003:** Build a lightweight pure Python execution environment mapping `Node` specs to Python/NumPy logic.
- [x][x][x] **Step 004:** Implement Python evaluation for `Add(Constant, Constant)`.
- [x][x][x] **Step 005:** Implement Python evaluation for `Sub(Constant, Constant)`.
- [x][x][x] **Step 006:** Implement Python evaluation for `Mul(Constant, Constant)`.
- [x][x][x] **Step 007:** Implement Python evaluation for `Div(Constant, Constant)`.
- [x][x][x] **Step 008:** Implement Python evaluation for `Pow(Constant, Constant)`.
- [x][x][x] **Step 009:** Implement Python evaluation for `Abs(Constant)`.
- [x][x][x] **Step 010:** Implement Python evaluation for `Exp(Constant)`.
- [x][x][x] **Step 011:** Implement Python evaluation for `Log(Constant)`.
- [x][x][x] **Step 012:** Implement Python evaluation for `Sqrt(Constant)`.
- [x][x][x] **Step 013:** Implement Python evaluation for `Cast(Constant)`.
- [x][x][x] **Step 014:** Implement Python evaluation for `Reshape(Constant, Constant_Shape)`.
- [x][x][x] **Step 015:** Implement Python evaluation for `Transpose(Constant)`.
- [x][x][x] **Step 016:** Implement Python evaluation for `Squeeze(Constant)`.
- [x][x][x] **Step 017:** Implement Python evaluation for `Unsqueeze(Constant)`.
- [x][x][x] **Step 018:** Implement Python evaluation for `Flatten(Constant)`.
- [x][x][x] **Step 019:** Implement Python evaluation for `Concat([Constant, ...])`.
- [x][x][x] **Step 020:** Implement Python evaluation for `Slice(Constant, ...)`.
- [x][x][x] **Step 021:** Implement Python evaluation for `Gather(Constant, Constant_Indices)`.
- [x][x][x] **Step 022:** Implement Python evaluation for `Shape(Constant)` -> replaces with a new 1D `Constant` tensor.
- [x][x][x] **Step 023:** Implement Python evaluation for `Size(Constant)` -> replaces with a scalar `Constant`.
- [x][x][x] **Step 024:** Implement Python evaluation for `NonZero(Constant)`.
- [x][x][x] **Step 025:** Implement logic to replace evaluated nodes with new `Constant` nodes in the `ir.Graph`.
- [x][x][x] **Step 026:** Ensure folded constants preserve the exact shape and data type of the original ONNX specification.
- [x][x][x] **Step 027:** Implement constant folding for scalar inputs broadcasted against tensors.
- [x][x][x] **Step 028:** Write recursive constant folding: if a node folds, immediately check its consumers.
- [x][x][x] **Step 029:** Implement a memory limit threshold (e.g., skip folding if the resulting constant > 10MB to avoid protobuf bloat).
- [x][x][x] **Step 030:** Implement partial constant folding (e.g., `Mul(X, 0)` -> `0`, `Mul(X, 1)` -> `X`).
- [x][x][x] **Step 031:** Implement identity removal: `Add(X, 0)` -> `X`.
- [x][x][x] **Step 032:** Implement identity removal: `Pow(X, 1)` -> `X`.
- [x][x][x] **Step 033:** Implement identity removal: `Div(X, 1)` -> `X`.
- [x][x][x] **Step 034:** Implement identity removal: `Reshape(X, shape(X))` -> `X`.
- [x][x][x] **Step 035:** Write 50+ unit tests asserting the exact mathematical output of the Python folding engine matches ONNX Runtime.
- [x][x][x] **Step 036:** Handle INT64 vs INT32 precision issues carefully when folding shape operations.
- [x][x][x] **Step 037:** Ensure `float16` and `bfloat16` constants are folded using float32 precision but cast back correctly.
- [x][x][x] **Step 038:** Implement logic to completely bypass folding for nodes with non-deterministic behavior (`RandomUniform`).
- [x][x][x] **Step 039:** Write comprehensive logging of which nodes were folded successfully.
- [x][x][x] **Step 040:** Finalize Phase 1 Constant Folding Engine.

### Phase 2: Dead-Code & Identity Elimination
- [x][x][x] **Step 041:** Create the base `DCEPass(GraphPass)` in `src/onnx9000/optimize/simplifier/passes/dce.py`.
- [x][x][x] **Step 042:** Implement a reverse topological traversal to identify nodes with zero consumers.
- [x][x][x] **Step 043:** Recursively remove dead nodes until the graph stabilizes.
- [x][x][x] **Step 044:** Ensure `graph.outputs` are correctly preserved and never eliminated.
- [x][x][x] **Step 045:** Implement unreferenced `initializer` (weights) removal.
- [x][x][x] **Step 046:** Implement unreferenced `value_info` removal.
- [x][x][x] **Step 047:** Remove disconnected subgraphs entirely.
- [x][x][x] **Step 048:** Create the base `IdentityEliminationPass`.
- [x][x][x] **Step 049:** Detect and remove explicit `Identity` nodes, rewiring inputs to consumers.
- [x][x][x] **Step 050:** Detect redundant `Cast(Cast(X, type1), type1)` chains.
- [x][x][x] **Step 051:** Detect redundant `Reshape(Reshape(X, s1), s2)` -> `Reshape(X, s2)`.
- [x][x][x] **Step 052:** Detect redundant `Transpose(Transpose(X, perm1), perm2)` -> fuse permutations.
- [x][x][x] **Step 053:** Detect `Squeeze` immediately followed by `Unsqueeze` on the same axis.
- [x][x][x] **Step 054:** Detect `Slice` operations covering the entire tensor dimension -> remove.
- [x][x][x] **Step 055:** Detect `Concat` operations with a single input -> remove.
- [x][x][x] **Step 056:** Write graph rewiring utility to safely update all consumer `input` names without breaking DAG topology.
- [x][x][x] **Step 057:** Handle edge cases where an eliminated node was part of the `graph.outputs` (must rewire the output pointer).
- [x][x][x] **Step 058:** Implement a loop that alternates Constant Folding -> Identity Elimination -> DCE until no changes occur.
- [x][x][x] **Step 059:** Write tests verifying identical output values before and after DCE.
- [x][x][x] **Step 060:** Test DCE on graphs with complex control flow (If/Loop subgraphs).
- [x][x][x] **Step 061:** Ensure side-effecting operations (if any exist) are protected from DCE.
- [x][x][x] **Step 062:** Finalize Phase 2 DCE & Identity.

### Phase 3: Operator Fusion (Algebraic Simplification)
- [x][x][x] **Step 063:** Create the base `FusionPass(GraphPass)` in `src/onnx9000/optimize/simplifier/passes/fusion.py`.
- [x][x][x] **Step 064:** Implement `Conv` + `BatchNorm` fusion algorithm in pure Python.
- [x][x][x] **Step 065:** Implement `ConvTranspose` + `BatchNorm` fusion.
- [x][x][x] **Step 066:** Implement `MatMul` + `Add` -> `Gemm` fusion.
- [x][x][x] **Step 067:** Implement `Gemm` + `BatchNorm` fusion.
- [x][x][x] **Step 068:** Implement `Conv` + `Add` (Bias) fusion.
- [x][x][x] **Step 069:** Implement `Conv` + `Mul` (Scaling) fusion.
- [x][x][x] **Step 070:** Implement `BatchNorm` + `Relu` fusion (custom op generation).
- [x][x][x] **Step 071:** Implement `MatMul` + `Relu` fusion.
- [x][x][x] **Step 072:** Implement LayerNorm fusion (Sub -> Pow -> ReduceMean -> Add -> Div -> Mul -> Add) -> `LayerNormalization`.
- [x][x][x] **Step 073:** Implement Gelu fusion (Div -> Erf -> Add -> Mul -> Mul) -> `Gelu`.
- [x][x][x] **Step 074:** Implement FastGelu/QuickGelu fusions.
- [x][x][x] **Step 075:** Implement Softmax fusion (Exp -> ReduceSum -> Div) -> `Softmax`.
- [x][x][x] **Step 076:** Implement Swish / SiLU fusion (Sigmoid -> Mul) -> `Swish`.
- [x][x][x] **Step 077:** Implement GroupNorm fusion pattern matching.
- [x][x][x] **Step 078:** Ensure `Conv` + `BatchNorm` fusion updates the `initializer` tensors mathematically (folding the variance/mean into the weights).
- [x][x][x] **Step 079:** Write rigorous tests asserting fused mathematical outputs exactly match unfused outputs.
- [x][x][x] **Step 080:** Handle edge cases where an intermediate tensor in a fusion pattern has external consumers (cannot fuse).
- [x][x][x] **Step 081:** Implement a pattern matching DSL (Domain Specific Language) in Python to define fusion rules easily.
- [x][x][x] **Step 082:** Ensure `Gemm` fusion correctly handles `transA` and `transB` attributes.
- [x][x][x] **Step 083:** Test fusion passes on a standard ResNet-50 PyTorch export.
- [x][x][x] **Step 084:** Test fusion passes on a standard Transformer (BERT/GPT) PyTorch export.
- [x][x][x] **Step 085:** Finalize Phase 3 Algebraic Fusion.

### Phase 4: Static Shape Inference & Validation
- [x][x][x] **Step 086:** Create the base `ShapeInferencePass(GraphPass)` natively in Python.
- [x][x][x] **Step 087:** Implement logic to propagate tensor shapes from inputs to outputs for all standard Opset 18 ops.
- [x][x][x] **Step 088:** Implement shape broadcasting rules accurately mirroring ONNX/NumPy.
- [x][x][x] **Step 089:** Implement logic to deduce output shapes for `Conv` based on `pads`, `strides`, `dilations`.
- [x][x][x] **Step 090:** Implement logic to deduce output shapes for `MaxPool`, `AveragePool`.
- [x][x][x] **Step 091:** Implement shape propagation for `MatMul` and `Gemm`.
- [x][x][x] **Step 092:** Implement shape propagation for `Reshape` (handling the `-1` dimension).
- [x][x][x] **Step 093:** Implement shape propagation for `Transpose`.
- [x][x][x] **Step 094:** Implement shape propagation for `Concat`, `Slice`, `Gather`.
- [x][x][x] **Step 095:** Store inferred shapes securely in the `value_info` field of the `ir.Graph`.
- [x][x][x] **Step 096:** Implement symbolic shape tracking (e.g., using strings like `'batch_size'`, `'seq_len'`).
- [x][x][x] **Step 097:** Implement an algebraic solver for symbolic shapes (e.g., `'seq_len' + 2`).
- [x][x][x] **Step 098:** Ensure every node in the simplified graph has completely defined shape information (critical for WebGPU static buffer allocation).
- [x][x][x] **Step 099:** Implement validation pass verifying topological order.
- [x][x][x] **Step 100:** Implement validation pass checking for DAG cycles.
- [x][x][x] **Step 101:** Implement validation pass checking for dangling inputs/outputs.
- [x][x][x] **Step 102:** Implement validation pass ensuring all `Node` attributes strictly match ONNX Schema types.
- [x][x][x] **Step 103:** Write tests comparing pure Python shape inference against `onnx.shape_inference` C++ outputs.
- [x][x][x] **Step 104:** Handle dynamic control flow shape inference (`If` / `Loop`).
- [x][x][x] **Step 105:** Finalize Phase 4 Shape Inference.

### Phase 5: High-Level API Integration
- [x][x][x] **Step 106:** Create the main `onnx9000.optimize.simplify(model)` entry point.
- [x][x][x] **Step 107:** Implement `skip_fusions` and `skip_constant_folding` configuration flags.
- [x][x][x] **Step 108:** Ensure `simplify()` runs fully within Pyodide without blocking the browser thread.
- [x][x][x] **Step 109:** Implement a progress callback during the `simplify` while-loop.
- [x][x][x] **Step 110:** Integrate `onnx9000-cli simplify input.onnx output.onnx` command line interface.
- [x][x][x] **Step 111:** Write an optimization report summarizing eliminated nodes, folded constants, and size reduction.
- [x][x][x] **Step 112:** Test `simplify` on >100 diverse models from the ONNX Model Zoo.
- [x][x][x] **Step 113:** Profile pure Python simplification latency (should be <5 seconds for typical models).
- [x][x][x] **Step 114:** Optimize the pattern matcher using regex-like DAG matching algorithms.
- [x][x][x] **Step 115:** Implement a 'dry run' to estimate simplification benefits without modifying the original graph.
- [x][x][x] **Step 116:** Support generating an HTML visual report comparing graphs.
- [x][x][x] **Step 117:** Implement memory mapping for parsing massive >2GB models before simplification.
- [x][x][x] **Step 118:** Publish `@onnx9000/simplifier` documentation.
- [x][x][x] **Step 119:** Finalize Phase 5 and the ONNX5 Simplifier Architecture.

