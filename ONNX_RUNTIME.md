# Exhaustive ONNX Runtime (ORT) Feature & API Checklist

This document tracks the progress towards implementing an execution engine that matches the features, APIs, and optimization capabilities of [Microsoft's ONNX Runtime](https://github.com/microsoft/onnxruntime).

**🔥 STATUS:** `onnx9000` is maturing rapidly. Over 200 standard ONNX operators have been implemented. The core IR parsing, autograd engine, static C++ memory planning, Apple Accelerate framework, WebAssembly SIMD backends, CUDA target, and control flow operators are fully integrated.


## 1. Core C API (`OrtApi` and Structs)

Implementing the `OrtApi` ABI guarantees plugin and binding compatibility.

### 1.1 Opaque Structs (Handles)

#### `OrtEnv`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


#### `OrtSessionOptions`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


#### `OrtSession`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


#### `OrtRunOptions`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


#### `OrtAllocator`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


#### `OrtMemoryInfo`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


#### `OrtTensorTypeAndShapeInfo`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


#### `OrtValue`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


#### `OrtModelMetadata`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


#### `OrtThreadingOptions`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


#### `OrtIoBinding`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


#### `OrtArenaCfg`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


#### `OrtCustomOpDomain`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


#### `OrtCustomOp`

- [x] [x] [x] Lifecycle Management (Create/Release)

- [x] [x] [x] Thread-safe reference counting (if applicable)

- [x] [x] [x] Internal state encapsulation


### 1.2 `OrtApi` Function Pointers

## 2. Execution Providers (EPs)

### CPUExecutionProvider

- [x] [x] [x] `IExecutionProvider` Interface implementation

- [x] [x] [x] `GetCapability` (Graph Partitioning logic)

- [x] [x] [x] `Compile` (Sub-graph compilation if EP-driven)

- [x] [x] [x] Kernel Registry mapping

- [x] [x] [x] Memory Allocator (`OrtMemoryInfo` integration)

- [x] [x] [x] Data Transfer (`IDataTransfer` for CPU <-> EP)

- [x] [x] [x] MLAS (Microsoft Linear Algebra Subprograms) integration

- [x] [x] [x] Intra/Inter thread pools



### CUDAExecutionProvider

- [x] [x] [x] `IExecutionProvider` Interface implementation

- [x] [x] [x] `GetCapability` (Graph Partitioning logic)

- [x] [x] [x] `Compile` (Sub-graph compilation if EP-driven)

- [x] [x] [x] Kernel Registry mapping

- [x] [x] [x] Memory Allocator (`OrtMemoryInfo` integration)

- [x] [x] [x] Data Transfer (`IDataTransfer` for CPU <-> EP)

- [x] [x] [x] CUDA Streams

- [x] [x] [x] CuDNN Convolution algorithms

- [x] [x] [x] Arena Allocator on Device

- [x] [x] [x] FP16/BF16 optimizations



### TensorrtExecutionProvider

- [x] [x] [x] `IExecutionProvider` Interface implementation

- [x] [x] [x] `GetCapability` (Graph Partitioning logic)

- [x] [x] [x] `Compile` (Sub-graph compilation if EP-driven)

- [x] [x] [x] Kernel Registry mapping

- [x] [x] [x] Memory Allocator (`OrtMemoryInfo` integration)

- [x] [x] [x] Data Transfer (`IDataTransfer` for CPU <-> EP)

- [x] [x] [x] TRT Builder configuration

- [x] [x] [x] Engine caching

- [x] [x] [x] Sub-graph partitioning

- [x] [x] [x] Dynamic shapes support



### OpenVINOExecutionProvider

- [x] [x] [x] `IExecutionProvider` Interface implementation

- [x] [x] [x] `GetCapability` (Graph Partitioning logic)

- [x] [x] [x] `Compile` (Sub-graph compilation if EP-driven)

- [x] [x] [x] Kernel Registry mapping

- [x] [x] [x] Memory Allocator (`OrtMemoryInfo` integration)

- [x] [x] [x] Data Transfer (`IDataTransfer` for CPU <-> EP)

- [x] [x] [x] NGraph IR translation

- [x] [x] [x] VPU/NPU device targeting

- [x] [x] [x] Compiled Model cache



### DmlExecutionProvider

- [x] [x] [x] `IExecutionProvider` Interface implementation

- [x] [x] [x] `GetCapability` (Graph Partitioning logic)

- [x] [x] [x] `Compile` (Sub-graph compilation if EP-driven)

- [x] [x] [x] Kernel Registry mapping

- [x] [x] [x] Memory Allocator (`OrtMemoryInfo` integration)

- [x] [x] [x] Data Transfer (`IDataTransfer` for CPU <-> EP)

- [x] [x] [x] DirectML device mapping

- [x] [x] [x] UAV/SRV barriers

- [x] [x] [x] Command List recording



### CoreMLExecutionProvider

- [x] [x] [x] `IExecutionProvider` Interface implementation

- [x] [x] [x] `GetCapability` (Graph Partitioning logic)

- [x] [x] [x] `Compile` (Sub-graph compilation if EP-driven)

- [x] [x] [x] Kernel Registry mapping

- [x] [x] [x] Memory Allocator (`OrtMemoryInfo` integration)

- [x] [x] [x] Data Transfer (`IDataTransfer` for CPU <-> EP)

- [x] [x] [x] ANE (Apple Neural Engine) targeting

- [x] [x] [x] CoreML model translation

- [x] [x] [x] NHWC layout translation



### XnnpackExecutionProvider

- [x] [x] [x] `IExecutionProvider` Interface implementation

- [x] [x] [x] `GetCapability` (Graph Partitioning logic)

- [x] [x] [x] `Compile` (Sub-graph compilation if EP-driven)

- [x] [x] [x] Kernel Registry mapping

- [x] [x] [x] Memory Allocator (`OrtMemoryInfo` integration)

- [x] [x] [x] Data Transfer (`IDataTransfer` for CPU <-> EP)

- [x] [x] [x] WebAssembly SIMD targeting

- [x] [x] [x] NHWC preferred layout

- [x] [x] [x] Sub-graph compilation



### NnapiExecutionProvider

- [x] [x] [x] `IExecutionProvider` Interface implementation

- [x] [x] [x] `GetCapability` (Graph Partitioning logic)

- [x] [x] [x] `Compile` (Sub-graph compilation if EP-driven)

- [x] [x] [x] Kernel Registry mapping

- [x] [x] [x] Memory Allocator (`OrtMemoryInfo` integration)

- [x] [x] [x] Data Transfer (`IDataTransfer` for CPU <-> EP)

- [x] [x] [x] Android NNAPI model translation

- [x] [x] [x] Device capability queries

- [x] [x] [x] AHardwareBuffer support



### QNNExecutionProvider

- [x] [x] [x] `IExecutionProvider` Interface implementation

- [x] [x] [x] `GetCapability` (Graph Partitioning logic)

- [x] [x] [x] `Compile` (Sub-graph compilation if EP-driven)

- [x] [x] [x] Kernel Registry mapping

- [x] [x] [x] Memory Allocator (`OrtMemoryInfo` integration)

- [x] [x] [x] Data Transfer (`IDataTransfer` for CPU <-> EP)

- [x] [x] [x] Qualcomm AI Engine Direct

- [x] [x] [x] HTP/DSP graph offloading

- [x] [x] [x] Context binary caching



### ROCmExecutionProvider

- [x] [x] [x] `IExecutionProvider` Interface implementation

- [x] [x] [x] `GetCapability` (Graph Partitioning logic)

- [x] [x] [x] `Compile` (Sub-graph compilation if EP-driven)

- [x] [x] [x] Kernel Registry mapping

- [x] [x] [x] Memory Allocator (`OrtMemoryInfo` integration)

- [x] [x] [x] Data Transfer (`IDataTransfer` for CPU <-> EP)

- [x] [x] [x] HIP Streams

- [x] [x] [x] MIOpen Conv algorithms

- [x] [x] [x] AMD GPU memory arenas



### MIGraphXExecutionProvider

- [x] [x] [x] `IExecutionProvider` Interface implementation

- [x] [x] [x] `GetCapability` (Graph Partitioning logic)

- [x] [x] [x] `Compile` (Sub-graph compilation if EP-driven)

- [x] [x] [x] Kernel Registry mapping

- [x] [x] [x] Memory Allocator (`OrtMemoryInfo` integration)

- [x] [x] [x] Data Transfer (`IDataTransfer` for CPU <-> EP)

- [x] [x] [x] MIGraphX graph compilation

- [x] [x] [x] AMDGPU specific fusions



### TvmExecutionProvider

- [x] [x] [x] `IExecutionProvider` Interface implementation

- [x] [x] [x] `GetCapability` (Graph Partitioning logic)

- [x] [x] [x] `Compile` (Sub-graph compilation if EP-driven)

- [x] [x] [x] Kernel Registry mapping

- [x] [x] [x] Memory Allocator (`OrtMemoryInfo` integration)

- [x] [x] [x] Data Transfer (`IDataTransfer` for CPU <-> EP)

- [x] [x] [x] TVM Relay translation

- [x] [x] [x] AutoTVM tuning cache loading



### WebNNExecutionProvider

- [x] [x] [x] `IExecutionProvider` Interface implementation

- [x] [x] [x] `GetCapability` (Graph Partitioning logic)

- [x] [x] [x] `Compile` (Sub-graph compilation if EP-driven)

- [x] [x] [x] Kernel Registry mapping

- [x] [x] [x] Memory Allocator (`OrtMemoryInfo` integration)

- [x] [x] [x] Data Transfer (`IDataTransfer` for CPU <-> EP)

- [x] [x] [x] WebNN API builder integration

- [x] [x] [x] Browser GPU/NPU dispatching



## 3. Graph Optimizations

Graph optimizations reduce kernel launches, memory footprint, and computation.

### Level 1 (Basic - Type/Shape/Constant)

#### `ConstantFolding`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `RedundantNodeElimination`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `CastElimination`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `IdentityElimination`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `UnsqueezeElimination`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `SliceElimination`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `DropoutElimination`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `NodeBreakFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `ShapeToInitializer`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `ReshapeFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `FreeDimensionOverride`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


### Level 2 (Extended - Fusions)

#### `ConvActivationFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `ConvBatchNormFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `ConvAddFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `MatMulAddFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `MatMulScaleFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `GemmActivationFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `LayerNormFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `SimplifiedLayerNormFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `AttentionFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `EmbedLayerNormFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `BiasGeluFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `FastGeluFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `SkipLayerNormFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `QLinearConvFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `QLinearMatMulFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `RotaryEmbeddingFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `MultiHeadAttentionFusion`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


### Level 3 (Layout Transformations)

#### `NCHW_to_NHWC_Transformation`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `NHWC_to_NCHW_Transformation`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


#### `NCDHW_to_NDHWC_Transformation`

- [x] [x] [x] Graph Pattern Matching

- [x] [x] [x] Node replacement strategy

- [x] [x] [x] Type and Shape re-inference post-mutation

- [x] [x] [x] EP Assignment awareness (only fuse if target EP supports it)


## 4. Session & Run Configurations

### 4.1 SessionOptions

- [x] [x] [x] Implement `IntraOpNumThreads`

- [x] [x] [x] Wire `IntraOpNumThreads` to internal Session State

- [x] [x] [x] Implement `InterOpNumThreads`

- [x] [x] [x] Wire `InterOpNumThreads` to internal Session State

- [x] [x] [x] Implement `GraphOptimizationLevel`

- [x] [x] [x] Wire `GraphOptimizationLevel` to internal Session State

- [x] [x] [x] Implement `OptimizedModelFilePath`

- [x] [x] [x] Wire `OptimizedModelFilePath` to internal Session State

- [x] [x] [x] Implement `ExecutionMode (Sequential vs Parallel)`

- [x] [x] [x] Wire `ExecutionMode (Sequential vs Parallel)` to internal Session State

- [x] [x] [x] Implement `LogId`

- [x] [x] [x] Wire `LogId` to internal Session State

- [x] [x] [x] Implement `LogSeverityLevel`

- [x] [x] [x] Wire `LogSeverityLevel` to internal Session State

- [x] [x] [x] Implement `LogVerbosityLevel`

- [x] [x] [x] Wire `LogVerbosityLevel` to internal Session State

- [x] [x] [x] Implement `CustomProfilers`

- [x] [x] [x] Wire `CustomProfilers` to internal Session State

- [x] [x] [x] Implement `EnableCpuMemArena`

- [x] [x] [x] Wire `EnableCpuMemArena` to internal Session State

- [x] [x] [x] Implement `EnableMemPattern`

- [x] [x] [x] Wire `EnableMemPattern` to internal Session State

- [x] [x] [x] Implement `ConfigEntries (Key/Value pairs)`

- [x] [x] [x] Wire `ConfigEntries (Key/Value pairs)` to internal Session State


### 4.2 RunOptions

- [x] [x] [x] Implement `RunLogSeverityLevel`

- [x] [x] [x] Wire `RunLogSeverityLevel` to Execution Frame

- [x] [x] [x] Implement `RunLogVerbosityLevel`

- [x] [x] [x] Wire `RunLogVerbosityLevel` to Execution Frame

- [x] [x] [x] Implement `RunTag`

- [x] [x] [x] Wire `RunTag` to Execution Frame

- [x] [x] [x] Implement `Terminate (Cancellation support)`

- [x] [x] [x] Wire `Terminate (Cancellation support)` to Execution Frame

- [x] [x] [x] Implement `OnlyExecutePathToFetches`

- [x] [x] [x] Wire `OnlyExecutePathToFetches` to Execution Frame


## 5. Memory Management

### `BFCArena (Best-Fit with Coalescing Allocator)`

- [x] [x] [x] Structural Definition

- [x] [x] [x] Allocation/Deallocation hooks

- [x] [x] [x] Lifecycle safety / Ref-counting


### `DeviceAllocator (Interface mapping to malloc/cudaMalloc etc)`

- [x] [x] [x] Structural Definition

- [x] [x] [x] Allocation/Deallocation hooks

- [x] [x] [x] Lifecycle safety / Ref-counting


### `OrtValue (Type-erased container for Tensors/Sequences/Maps)`

- [x] [x] [x] Structural Definition

- [x] [x] [x] Allocation/Deallocation hooks

- [x] [x] [x] Lifecycle safety / Ref-counting


### `Tensor (Dense array with shape/type)`

- [x] [x] [x] Structural Definition

- [x] [x] [x] Allocation/Deallocation hooks

- [x] [x] [x] Lifecycle safety / Ref-counting


### `SparseTensor (COO, CSR, BlockSparse formats)`

- [x] [x] [x] Structural Definition

- [x] [x] [x] Allocation/Deallocation hooks

- [x] [x] [x] Lifecycle safety / Ref-counting


### `IAllocator (Abstract interface)`

- [x] [x] [x] Structural Definition

- [x] [x] [x] Allocation/Deallocation hooks

- [x] [x] [x] Lifecycle safety / Ref-counting


### `MemoryPatternPlanner (Pre-computes memory offsets for statically sized graphs)`

- [x] [x] [x] Structural Definition

- [x] [x] [x] Allocation/Deallocation hooks

- [x] [x] [x] Lifecycle safety / Ref-counting


## 6. Language Bindings Wrapper Parity

### C++ (`onnxruntime_cxx_api.h`)

#### `Env`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `Session`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `SessionOptions`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `RunOptions`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `Value`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `ModelMetadata`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `IoBinding`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `MemoryInfo`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `AllocatorWithDefaultOptions`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


### Python (`onnxruntime` package)

#### `InferenceSession`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `SessionOptions`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `RunOptions`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `OrtValue`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `IOBinding`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `get_device()`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `get_available_providers()`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


### C# (`Microsoft.ML.OnnxRuntime`)

#### `InferenceSession`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `SessionOptions`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `RunOptions`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `OrtValue`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `NamedOnnxValue`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `Disposable patterns`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


### Java (`ai.onnxruntime`)

#### `OrtEnvironment`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `OrtSession`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `OnnxTensor`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `OnnxSequence`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `OnnxMap`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `JNI Bridges`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


### JavaScript (`onnxruntime-web` / `onnxruntime-node`)

#### `InferenceSession`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `Tensor`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `Env configurations`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `WASM threads/SIMD flags`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


#### `WebGL/WebGPU backend mapping`

- [x] [x] [x] Binding interface implemented

- [x] [x] [x] Memory lifecycle bridged correctly (avoiding leaks)

- [x] [x] [x] Error propagation mapping (C OrtStatus to Language Exception)


## 7. Quantization Tooling

### 7.1 Static Quantization (Calibration)

- [x] [x] [x] MinMax Calibration

- [x] [x] [x] Entropy Calibration (KL Divergence)

- [x] [x] [x] Percentile Calibration

### 7.2 Dynamic Quantization

- [x] [x] [x] Weight-only integer cast

- [x] [x] [x] Runtime activation symmetric/asymmetric quantization

### 7.3 Formats Support

- [x] [x] [x] QDQ (QuantizeLinear/DequantizeLinear) processing

- [x] [x] [x] QOperator (QLinearConv, QLinearMatMul) processing


## 8. Custom Operators

### `OrtCustomOpDomain`

- [x] [x] [x] Domain registration

- [x] [x] [x] Version tracking

### `OrtCustomOp`

- [x] [x] [x] `GetName` / `GetExecutionProviderType`

- [x] [x] [x] `GetInputType` / `GetOutputType`

- [x] [x] [x] `CreateKernel` / `Compute`

- [x] [x] [x] Safe execution context (`OrtKernelContext` wrapper)


## 9. Advanced Execution Mechanics

### IOBinding (Pre-binding inputs/outputs to specific devices to avoid copies)

- [x] [x] [x] Architectural design mapped

- [x] [x] [x] Integration with Graph execution

- [x] [x] [x] Profiling hooks added


### Threadpool Management (Eigen Threadpool / ORT custom spinlocks)

- [x] [x] [x] Architectural design mapped

- [x] [x] [x] Integration with Graph execution

- [x] [x] [x] Profiling hooks added


### ExecutionFrame (Graph runtime state during a single `Run` call)

- [x] [x] [x] Architectural design mapped

- [x] [x] [x] Integration with Graph execution

- [x] [x] [x] Profiling hooks added


### SessionState (Pre-allocated buffers, initialized weights, EP maps)

- [x] [x] [x] Architectural design mapped

- [x] [x] [x] Integration with Graph execution

- [x] [x] [x] Profiling hooks added


### DataTransferManager (Routing tensors across CPU/GPU boundaries)

- [x] [x] [x] Architectural design mapped

- [x] [x] [x] Integration with Graph execution

- [x] [x] [x] Profiling hooks added


## 10. Training APIs (`onnxruntime-training`)

### 10.1 Backend Graph Builders

- [x] [x] [x] Gradient Graph Builder (Auto-differentiation mapping)

- [x] [x] [x] Loss node insertion

- [x] [x] [x] Optimizer node insertion (AdamW, Lamb, SGD)

### 10.2 Frontend Integration

- [x] [x] [x] `ORTModule` (PyTorch `torch.nn.Module` interceptor)

- [x] [x] [x] Checkpoint Loading/Saving API

- [x] [x] [x] ATen operator translation bridge

## 11. Integration with ml-switcheroo
This section tracks the end-to-end integration of the onnx9000 engine within the parent `ml-switcheroo` project ecosystem.

### 11.1 IR to ONNX Export
- [x] [x] [x] Translate `ml-switcheroo` internal IR nodes to standard ONNX operators.
- [x] [x] [x] Correctly map internal data types to ONNX `TensorProto` data types.
- [x] [x] [x] Handle dynamic shapes and broadcast semantics during translation.
- [x] [x] [x] Serialize the compiled graph into a valid ONNX `ModelProto` binary.

### 11.2 In-Browser Training & Serving Pipeline
- [x] [x] [x] **Training in the Browser:**
  - [x] [x] [x] Implement/Integrate training graph builder compatible with `onnxruntime-web` training APIs.
  - [x] [x] [x] Support inserting loss functions and optimizers into the exported ONNX model.
  - [x] [x] [x] Manage WebAssembly (WASM) / WebGPU memory efficiently during training iterations.
- [x] [x] [x] **Serving in the Browser:**
  - [x] [x] [x] Seamlessly transition the trained model state (updated weights) to an inference session.
  - [x] [x] [x] Perform fast, zero-copy (where possible) browser-based inference.

### 11.3 External Pipeline Interoperability
- [x] [x] [x] **Download ONNX Feature:**
  - [x] [x] [x] Expose an API/UI button to download the strictly compliant `.onnx` model file.
- [x] [x] [x] **Official Pipeline Compatibility:**
  - [x] [x] [x] Verify the downloaded model runs correctly on standard `onnxruntime` (Python/C++).
  - [x] [x] [x] Verify the downloaded model can be used with official ONNX training servers (e.g., ORTModule/PyTorch).
