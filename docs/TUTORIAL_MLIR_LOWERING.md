# Understanding the `onnx9000` MLIR Lowering Pipeline

The compilation from an ONNX model into WebGPU shader code or WebAssembly bytecode undergoes a multi-stage lowering process.

1. **ONNX to MHLO (High-Level Dialect):**
   - Operations like `Add`, `Conv`, and `MatMul` are mapped to target-agnostic MHLO operations with resolved data types and shapes.
2. **MHLO to Linalg (Structural Dialect):**
   - Implicit looping patterns are expanded into explicit `linalg.generic` iterations. Elementwise fusion occurs here.
3. **Bufferization:**
   - Value semantics (`Tensor`) are lowered into memory semantics (`MemRef`). Allocations are explicitly scheduled.
4. **Linalg to HAL (Hardware Abstraction Layer):**
   - Nested loops are compiled into native executables. Command buffers are built to schedule dispatches and copy operations.
5. **HAL to VM (Virtual Machine Control Flow):**
   - The command buffer sequence is flattened into bytecode. Dynamic shapes and symbolic variables are bound. Register allocation maps SSA definitions into a small flat array of VM registers.

This multi-layer architecture enables retargeting the entire pipeline, from compiling to `.wvm` bytecode for pure JS interpreters, or emitting a fully self-contained Standalone JS payload using WGSL shaders string literals.
