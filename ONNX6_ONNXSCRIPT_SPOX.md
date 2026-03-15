# ONNX6: ONNXScript / Spox Native Rewrite

## Introduction
**Target Projects:** [ONNXScript](https://github.com/microsoft/onnxscript) & [Spox](https://github.com/Quantco/spox)
**New Home:** `src/onnx9000/script/`

Both ONNXScript and Spox are powerful authoring tools designed to write ONNX models via Python functions and a fluent API. However, they inherently rely on the standard Google `protobuf` Python library (which is backed by a heavy C++ implementation for performance) and tightly couple themselves to the standard ONNX tooling infrastructure. This makes them brittle to compile and execute directly within strict, zero-dependency environments like browser WASM (Pyodide).

**The `onnx9000` Vision:** We are building a fluent, dependency-free Python API for dynamically authoring and modifying ONNX subgraphs and custom operators directly. By using our own lightweight, web-friendly `onnx_pb2.py` parser and `core.ir` representations, developers can explicitly construct ONNX graphs node-by-node or via Python decorators (`@onnx9000.script`) without ever invoking the heavy C++ `protobuf` library. This is crucial for dynamically assembling models on the client side (e.g., stitching together a tokenizer subgraph with an LLM subgraph inside the browser).

## Exhaustive Implementation Checklist (300+ Items)

### Phase 1: Core Fluent Authoring API (`onnx9000.script.op`)
- [x] **Step 001:** Create the base `onnx9000.script.op` namespace.
- [x] **Step 002:** Implement a dynamic `__getattr__` on the `op` namespace to lazily generate Node builders for any standard ONNX operation.
- [x] **Step 003:** Implement `op.Add(A, B)` returning a symbolic `Var` representing the output tensor.
- [x] **Step 004:** Implement `op.Sub(A, B)` returning a symbolic `Var`.
- [x] **Step 005:** Implement `op.Mul(A, B)` returning a symbolic `Var`.
- [x] **Step 006:** Implement `op.Div(A, B)` returning a symbolic `Var`.
- [x] **Step 007:** Implement `op.MatMul(A, B)` returning a symbolic `Var`.
- [x] **Step 008:** Implement `op.Relu(X)` returning a symbolic `Var`.
- [x] **Step 009:** Implement `op.Sigmoid(X)` returning a symbolic `Var`.
- [x] **Step 010:** Implement `op.Transpose(X, perm=[...])` supporting keyword attribute arguments.
- [x] **Step 011:** Implement `op.Reshape(data, shape, allowzero=0)` supporting keyword attribute arguments.
- [x] **Step 012:** Implement `op.Conv(X, W, B, pads=[...], strides=[...])` supporting keyword attribute arguments.
- [x] **Step 013:** Implement type checking for all keyword attributes (e.g., ensuring `pads` is `INTS`).
- [x] **Step 014:** Implement implicit type casting for Python scalars (e.g., `op.Add(X, 1)` automatically creates a `Constant` node for `1`).
- [x] **Step 015:** Implement implicit list-to-tensor casting for array attributes (e.g., `op.Concat([A, B], axis=0)`).
- [x] **Step 016:** Implement a `Var` class representing edges (tensors) connecting nodes.
- [x] **Step 017:** Implement operator overloading on `Var` objects (`A + B` -> `op.Add(A, B)`).
- [x] **Step 018:** Implement operator overloading (`-`, `*`, `/`, `**`, `@`) on `Var` objects.
- [x] **Step 019:** Implement operator overloading (`>`, `<`, `==`, `!=`) on `Var` objects mapping to ONNX logical ops.
- [x] **Step 020:** Implement operator overloading (`&`, `|`, `~`) on `Var` objects mapping to ONNX bitwise ops.
- [x] **Step 021:** Implement slicing (`A[:, 1:3]`) on `Var` objects mapping to `op.Slice`.
- [x] **Step 022:** Implement indexing (`A[0]`) on `Var` objects mapping to `op.Gather` or `op.GatherElements`.
- [x] **Step 023:** Write logic to generate unique, deterministic string names for every intermediate `Var`.
- [x] **Step 024:** Support explicitly naming a `Var` (e.g., `A = op.Add(X, Y).rename('my_tensor')`).
- [x] **Step 025:** Implement `op.Constant(value=np.array([...]))` supporting NumPy arrays.
- [x] **Step 026:** Implement `op.Constant(value=...)` supporting pure Python nested lists and types.
- [x] **Step 027:** Ensure `op.Constant` correctly serializes floats to ONNX `FLOAT` and ints to ONNX `INT64`.
- [x] **Step 028:** Implement multi-output node handling (e.g., `values, indices = op.TopK(X, K)`).
- [x] **Step 029:** Support unpacking multi-output tuples from the authoring API.
- [x] **Step 030:** Write 50+ unit tests verifying the exact `ir.Node` structures produced by the `op` namespace match standard ONNX specifications.
- [x] **Step 031:** Implement strict schema validation during `op.XXX()` calls (validating input counts and types against Opset 18+).
- [x] **Step 032:** Provide informative `TypeError` or `ValueError` exceptions if a user provides invalid attributes (e.g., passing a float to an `INTS` attribute).
- [x] **Step 033:** Implement a fallback for custom operations: `op.custom_domain.MyOp(X, attr1=...)`.
- [x] **Step 034:** Finalize Phase 1 Core Authoring API.

### Phase 2: `@onnx9000.script` Decorator & AST Parsing
- [x] **Step 035:** Implement the `@onnx9000.script` decorator.
- [x] **Step 036:** Implement standard Python `ast.parse` logic on the decorated function's source code.
- [x] **Step 037:** Translate `ast.FunctionDef` arguments into ONNX `Graph` inputs.
- [x] **Step 038:** Translate `ast.Return` statements into ONNX `Graph` outputs.
- [x] **Step 039:** Translate `ast.Assign` (e.g., `Y = op.Add(X, Z)`) into ONNX nodes.
- [x] **Step 040:** Translate `ast.If` blocks into ONNX `If` subgraphs (requires creating `IfBranch` subgraphs natively).
- [x] **Step 041:** Ensure variables defined inside `ast.If` branches are correctly exported to the parent scope.
- [x] **Step 042:** Translate `ast.For` blocks over tensors into ONNX `Loop` or `Scan` nodes.
- [x] **Step 043:** Translate `ast.While` blocks into ONNX `Loop` nodes.
- [x] **Step 044:** Capture closed-over variables from the global scope and embed them as `Constant` nodes.
- [x] **Step 045:** Support tracing standard Python `if/else` (when the condition is a static boolean, unroll it; when it's a Tensor, build an ONNX `If`).
- [x] **Step 046:** Implement comprehensive error reporting mapping AST parse errors back to the exact line number of the user's Python code.
- [x] **Step 047:** Support calling other `@onnx9000.script` decorated functions from within a script function (inlining the graph).
- [x] **Step 048:** Implement logic to treat `@onnx9000.script` functions as custom ONNX `FunctionProto` definitions instead of inlining them.
- [x] **Step 049:** Write tests verifying complex nested `If` and `Loop` AST parsing exactly matches ONNX control flow semantics.
- [x] **Step 050:** Ensure `ast.ListComp` (list comprehensions) are either unrolled statically or rejected with a clear error.
- [x] **Step 051:** Handle Python multiple assignment (`a, b = op.Split(...)`).
- [x] **Step 052:** Support type hints (`def my_model(x: Float[10, 20]) -> Float[10, 20]:`) to enforce ONNX `TypeProto` generation.
- [x] **Step 053:** Extract shape information from type hints directly into the generated `Graph` `value_info`.
- [x] **Step 054:** Finalize Phase 2 AST Parsing logic.

### Phase 3: Graph Builder & Top-Down Construction
- [x] **Step 055:** Implement a `GraphBuilder` class in `src/onnx9000/script/builder.py`.
- [x] **Step 056:** Implement `builder.add_input(name, dtype, shape)`.
- [x] **Step 057:** Implement `builder.add_output(var, name)`.
- [x] **Step 058:** Implement `builder.add_initializer(name, array)`.
- [x] **Step 059:** Allow users to manually append `ir.Node` objects to the builder.
- [x] **Step 060:** Implement `builder.build()` to finalize and topologically sort the graph.
- [x] **Step 061:** Implement `builder.to_onnx()` to serialize to `ModelProto`.
- [x] **Step 062:** Implement `builder.merge(other_builder)` to combine two disjoint subgraphs.
- [x] **Step 063:** Implement an API to locate a specific node by name (`builder.get_node('Relu_1')`).
- [x] **Step 064:** Implement an API to replace a node with another node (`builder.replace(old_node, new_node)`).
- [x] **Step 065:** Implement an API to delete a node (`builder.delete(node)`).
- [x] **Step 066:** Implement an API to rewire edges (`builder.replace_input(node, old_var, new_var)`).
- [x] **Step 067:** Implement subgraph extraction: `builder.extract_subgraph(inputs=[...], outputs=[...])`.
- [x] **Step 068:** Ensure `extract_subgraph` performs dead-code elimination to prune unused branches.
- [x] **Step 069:** Write a utility to rename all nodes and tensors in a graph with a specific prefix (e.g., for namespace isolation during merging).
- [x] **Step 070:** Support importing an existing `.onnx` file into a `GraphBuilder` for editing.
- [x] **Step 071:** Write rigorous tests verifying `GraphBuilder` output is deterministic.
- [x] **Step 072:** Implement memory optimizations during graph building (avoiding duplicate string allocations for thousands of intermediate names).
- [x] **Step 073:** Support inserting custom metadata (`doc_string`, `domain`, `version`) into the finalized graph.
- [x] **Step 074:** Finalize Phase 3 Graph Building API.

### Phase 4: Explicit Control Flow Construction
- [x] **Step 075:** Create an explicit API for constructing ONNX `If` nodes: `op.If(condition, then_branch=..., else_branch=...)`.
- [x] **Step 076:** Create an explicit API for ONNX `Loop` nodes: `op.Loop(max_trip_count, cond, body=...)`.
- [x] **Step 077:** Create an explicit API for ONNX `Scan` nodes.
- [x] **Step 078:** Implement context managers for authoring subgraphs block-by-block (e.g., `with builder.If(cond): ...`).
- [x] **Step 079:** Handle implicit variable capture correctly when defining subgraphs (inputs from outer scope must become explicit graph inputs).
- [x] **Step 080:** Handle output propagation from subgraphs to the parent scope accurately.
- [x] **Step 081:** Write tests verifying `If` nodes correctly unify output types and shapes from both branches.
- [x] **Step 082:** Implement pure Python shape inference that penetrates into control flow subgraphs.
- [x] **Step 083:** Implement DCE that operates inside control flow subgraphs.
- [x] **Step 084:** Write a script that successfully constructs a complex Recurrent Neural Network (RNN) using only `op.Loop` and `op.MatMul`.
- [x] **Step 085:** Finalize Phase 4 Control Flow API.

### Phase 5: Opset Versioning & Schema Validation
- [x] **Step 086:** Create a strict schema registry mapping standard ONNX Opset 18+ operator constraints in pure Python.
- [x] **Step 087:** Implement `op.set_target_opset(version=18)` to configure the authoring environment.
- [x] **Step 088:** Provide informative warnings or errors if a user calls an `op.XXX` method that does not exist in the target opset.
- [x] **Step 089:** Implement automatic attribute upcasting/downcasting between Opsets (e.g., `axes` becoming inputs in later opsets).
- [x] **Step 090:** Implement type-checking validating that output tensor types match the ONNX specification for the given operation.
- [x] **Step 091:** Support exporting a graph containing custom opsets with a clearly defined `opset_import` array.
- [x] **Step 092:** Implement logic to parse external ONNX schema definitions (JSON/Protobuf) to auto-generate the `op.XXX` namespace dynamically.
- [x] **Step 093:** Write documentation detailing the subset of ONNX ops strictly supported by the `onnx9000` WebGPU backend.
- [x] **Step 094:** Implement a validation pass (`builder.validate()`) matching the rigor of standard `onnx.checker` entirely in pure Python.
- [x] **Step 095:** Test `builder.validate()` on deliberately malformed graphs (e.g., cyclic dependencies, mismatched tensor shapes).
- [x] **Step 096:** Finalize Phase 5 Opset & Schema Management.

### Phase 6: Pyodide Integration & Polish
- [x] **Step 097:** Ensure `import onnx9000.script` works flawlessly inside Pyodide.
- [x] **Step 098:** Write a JS wrapper `onnx9000.GraphBuilder` exposing the Python API to JS developers via Pyodide.
- [x] **Step 099:** Test dynamic generation of a simple 2-layer MLP purely in the browser.
- [x] **Step 100:** Test downloading the browser-generated MLP as a valid `.onnx` file.
- [x] **Step 101:** Implement syntax highlighting snippets for `onnx9000.script` for standard editors (VSCode).
- [x] **Step 102:** Provide extensive inline docstrings for every `op.XXX` method to enable IDE auto-completion.
- [x] **Step 103:** Write tutorial notebooks demonstrating how to build Custom Operators using `onnx9000.script`.
- [x] **Step 104:** Publish `@onnx9000/script` documentation.
- [x] **Step 105:** Finalize Phase 6 and the ONNX6 Scripting Architecture.

