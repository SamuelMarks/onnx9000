# onnx-tool Replication & Parity Tracker

## Description
This document tracks the complete reimplementation of `onnx-tool` within the `onnx9000` ecosystem.
The original `onnx-tool` is an excellent diagnostic utility for profiling MACs, FLOPs, parameter counts, and static memory allocations. However, it often relies on heavy system environments or is used purely as a CLI script.
Our `onnx9000` reimplementation integrates these advanced profiling and symbolic shape inference capabilities natively into our pure-Python, WASM-compatible Intermediate Representation. This means you can profile the exact memory bounds and compute intensity of a multi-GB transformer or vision model instantly in the browser or on a cold serverless node without executing the model or installing `onnxruntime`.

## Exhaustive Parity Checklist

### 1. Shape Inference & Symbolic Math (50+ items)
- [ ] Implement zero-dependency static shape inference
- [ ] Implement symbolic shape inference (e.g., tracking `batch_size`, `seq_len`)
- [ ] Support dynamic symbolic variables recursively across subgraphs
- [ ] Resolve symbolic math natively (e.g., `seq_len * 2`)
- [ ] Evaluate `Reshape` dynamic dimensions (`-1`) via volume preservation
- [ ] Evaluate `MatMul` output shapes symbolically `[..., M, K] x [..., K, N] -> [..., M, N]`
- [ ] Evaluate `Conv` output shapes statically (with strides, dilations, pads)
- [ ] Evaluate `ConvTranspose` output shapes statically
- [ ] Evaluate `MaxPool` output shapes statically
- [ ] Evaluate `AveragePool` output shapes statically
- [ ] Evaluate `GlobalAveragePool` output shapes statically
- [ ] Evaluate `Gather` output shapes dynamically based on indices length
- [ ] Evaluate `Slice` output shapes statically (when bounds are constant)
- [ ] Evaluate `Slice` output shapes symbolically (when bounds are dynamic variables)
- [ ] Evaluate `Concat` output shapes (summing across specified axis)
- [ ] Evaluate `Split` output shapes
- [ ] Evaluate `Tile` output shapes
- [ ] Evaluate `Expand` output shapes (broadcasting rules)
- [ ] Evaluate `Pad` output shapes (constant padding values)
- [ ] Evaluate `TopK` output shapes
- [ ] Evaluate `ArgMax` / `ArgMin` output shapes
- [ ] Evaluate `NonZero` output shapes (as fully dynamic / undefined bounds)
- [ ] Evaluate `Where` output shapes (broadcasting rules)
- [ ] Evaluate `Shape` node output values symbolically
- [ ] Evaluate `Size` node output values symbolically
- [ ] Handle explicit PyTorch symbolic naming (e.g., `SymInt`)
- [ ] Handle implicit broadcasting rules across all elementwise arithmetic (`Add`, `Mul`, `Div`, `Sub`)
- [ ] Handle implicit broadcasting rules across all logical ops (`And`, `Or`, `Equal`, `Less`)
- [ ] Propagate shape inference deeply through `If` subgraphs (ensuring branch shape equality)
- [ ] Propagate shape inference deeply through `Loop` subgraphs (resolving loop state dimensions)
- [ ] Validate loop body outputs match loop body inputs dimensionality
- [ ] Extract sequence length bounds dynamically for `SequenceConstruct`
- [ ] Extract sequence length bounds dynamically for `SplitToSequence`
- [ ] Implement custom shape inference for recognized custom operators (e.g., FlashAttention)
- [ ] Fallback gracefully when encountering unrecognized CustomOps (marking outputs as `Unknown`)
- [ ] Infer `Cast` output data types
- [ ] Propagate `dtype` inference alongside shape inference explicitly
- [ ] Handle type promotion rules strictly across math ops (e.g. Int32 + Float32 -> Float32)
- [ ] Handle float16 / bfloat16 propagation safely
- [ ] Deduplicate symbolically equivalent shapes (e.g. `dim0` vs `batch`)
- [ ] Allow manual user overrides for specific symbolic dimensions (e.g. `batch=1, seq=128`)
- [ ] Strip undefined dimensions to bounded sizes for profiling
- [ ] Track minimum bounds for dynamic shapes
- [ ] Track maximum bounds for dynamic shapes
- [ ] Inject `ValueInfo` natively back into the Graph after inference
- [ ] Remove explicitly redundant `ValueInfo` metadata to save space
- [ ] Throw explicit exceptions on shape validation failures (e.g. non-broadcastable adds)
- [ ] Support `Shape` operator constant folding via symbolic evaluation
- [ ] Support `Reshape` -> `Reshape` cancellations mathematically

### 2. MACs & FLOPs Computation Profiling (40+ items)
- [ ] Profile MACs (Multiply-Accumulates) for `MatMul`
- [ ] Profile FLOPs (Floating-Point Operations) for `MatMul`
- [ ] Profile MACs for `Conv` (Standard 2D/3D)
- [ ] Profile MACs for `Conv` (Depthwise / Grouped)
- [ ] Profile FLOPs for `Conv`
- [ ] Profile MACs for `ConvTranspose`
- [ ] Profile FLOPs for `ConvTranspose`
- [ ] Profile MACs for `Gemm`
- [ ] Profile FLOPs for `Gemm`
- [ ] Profile FLOPs for `BatchNormalization`
- [ ] Profile FLOPs for `LayerNormalization`
- [ ] Profile FLOPs for `InstanceNormalization`
- [ ] Profile FLOPs for Elementwise Math (`Add`, `Sub`, `Mul`, `Div`)
- [ ] Profile FLOPs for Transcendental Math (`Exp`, `Log`, `Sin`, `Cos`)
- [ ] Profile FLOPs for Activations (`Relu`, `Sigmoid`, `Tanh`, `Gelu`)
- [ ] Profile FLOPs for `Softmax`
- [ ] Profile FLOPs for `ReduceMean`, `ReduceSum`, `ReduceMax`, `ReduceMin`
- [ ] Profile Memory Bandwidth limits (Bytes Read/Written) for memory-bound ops (`Reshape`, `Transpose`)
- [ ] Support profiling dynamic shapes via symbolic mathematical formulas (`MACs = batch * seq * 1024`)
- [ ] Support profiling dynamic shapes via explicit overrides (`MACs(batch=1)`)
- [ ] Aggregate Total MACs globally
- [ ] Aggregate Total FLOPs globally
- [ ] Output per-node MACs summary
- [ ] Output per-node FLOPs summary
- [ ] Handle dynamic branching: average MACs across `If` subgraph branches
- [ ] Handle dynamic branching: worst-case MACs across `If` subgraph branches
- [ ] Handle dynamic looping: multiply loop body MACs by static loop iterations
- [ ] Handle dynamic looping: symbolic loop iterations (`MACs = N * body`)
- [ ] Profile Transformer architectures cleanly (Attention FLOPs tracking)
- [ ] Profile CNN architectures cleanly (spatial dimensionality tracking)
- [ ] Profile CustomOps (if custom FLOPs formula provided)
- [ ] Distinguish INT8 MACs vs FP32 MACs natively
- [ ] Distinguish INT4 MACs natively
- [ ] Distinguish FP16/BF16 MACs natively
- [ ] Ignore FLOPs for routing/index ops (`Gather`, `Scatter`, `NonZero`)
- [ ] Account for sparsity automatically in `SparseTensor` profiled ops (if structural)
- [ ] Print top-K most compute-intensive nodes (Bottleneck analysis)
- [ ] Print MACs/FLOPs distribution pie-chart data points
- [ ] Expose API to query cumulative FLOPs up to a specific node layer
- [ ] Provide ratio of Compute vs Memory-bound characteristics per node

### 3. Static Parameter & Memory Footprint Profiling (40+ items)
- [ ] Profile Total Parameter count
- [ ] Profile Total Constant Memory footprint (MB/GB)
- [ ] Distinguish trainable parameters (`Parameter` inputs) vs frozen constants
- [ ] Provide precise memory sizes based on `dtype` (FP32=4B, FP16=2B, INT8=1B)
- [ ] Profile activation memory footprint (peak memory during inference)
- [ ] Estimate working-set RAM requirement statically via topological simulation
- [ ] Calculate activation lifecycle boundaries (when tensors can be freed/reused)
- [ ] Simulate greedy memory arena allocation natively (calculating exact contiguous buffers)
- [ ] Simulate offset-based static memory allocation (for edge device targeting)
- [ ] Support dynamic shape overrides for activation footprint calculations
- [ ] Profile individual node memory ingestion (Input tensor sizes)
- [ ] Profile individual node memory generation (Output tensor sizes)
- [ ] Map parameter counts to specific sub-architectures (e.g., `layer1.attention` has X parameters)
- [ ] Identify redundant constant sizes (`Tile` -> `Constant` bloat detection)
- [ ] Track memory fragmentation in the simulated arena
- [ ] Identify shared initializers directly
- [ ] Ignore zero-size arrays natively
- [ ] Calculate total disk-size footprint vs RAM-size footprint (e.g. sparse formats vs dense expansion)
- [ ] Highlight un-fused nodes leading to excessive activation memory (e.g. `Conv` + `Add` separate tensors)
- [ ] Output layer-by-layer memory trajectory graph data
- [ ] Extract peak memory bottleneck node specifically
- [ ] Analyze attention mask activation memory (quadratic expansion profiling)
- [ ] Evaluate grouped convolution memory reductions
- [ ] Profile ONNX Sequence memory usage
- [ ] Support profiling specific INT4 packed weight models
- [ ] Provide precise memory savings report comparing FP32 -> FP16 -> INT8
- [ ] Check external data alignment requirements natively
- [ ] Detect huge attributes that should be Initializers
- [ ] Validate `Float64` usage to suggest downcasting for memory savings
- [ ] Validate `Int64` usage to suggest downcasting
- [ ] Aggregate overall "Model Compute Intensity" (FLOPs / Byte Ratio)
- [ ] Output pie-chart data points for Parameter distribution
- [ ] Output pie-chart data points for Activation distribution
- [ ] Provide heuristic bounds for WebGPU buffer limits (128MB max) vs Graph structure
- [ ] Expose API to get exact byte-offset of any tensor in a simulated arena
- [ ] Report estimated latency given theoretical hardware TOPS (TeraOps per second)
- [ ] Report estimated latency given theoretical hardware memory bandwidth (GB/s)

### 4. Graph Topology Optimization Checks (30+ items)
- [ ] Detect missing `ConstantFolding` opportunities automatically
- [ ] Detect redundant `Transpose` operations natively
- [ ] Detect redundant `Cast` operations
- [ ] Detect missing `BatchNorm` fusion opportunities
- [ ] Detect missing `Scale` fusion opportunities
- [ ] Detect missing `Gelu` fusion opportunities
- [ ] Detect un-fused `MatMul` + `Add` structures
- [ ] Identify deeply nested `If` subgraphs that can be flattened
- [ ] Identify scalar math chains that can be analytically simplified
- [ ] Highlight completely unused initializers
- [ ] Highlight completely unused global inputs
- [ ] Detect Identity/No-Op chains
- [ ] Analyze sparsity of weight constants to suggest pruning optimizations
- [ ] Suggest `int4` quantization if weight distributions are highly uniform
- [ ] Detect dynamic ops (`NonZero`) driving massive downstream dynamic allocations
- [ ] Suggest replacing older ONNX constructs with modern variants (Opset suggestions)
- [ ] Flag operations known to be slow on GPUs (e.g., dynamic `Loop`)
- [ ] Flag operations unsupported by common WebGPU backends (e.g., complex numbers)
- [ ] Profile tree-ensemble transpilation complexity statically
- [ ] Generate detailed "Optimization Opportunities" text report
- [ ] Highlight layout conflicts (mixing NCHW and NHWC implicitly)
- [ ] Highlight data type bottlenecks (e.g., FP16 -> FP32 -> FP16 sandwiches)
- [ ] Expose programmatic JSON list of all identified optimizations
- [ ] Expose automated apply functions for the identified optimizations (via GraphSurgeon)

### 5. Detailed Layer/Module Analysis & Grouping (30+ items)
- [ ] Implement smart node-grouping based on naming conventions (e.g., `model.layer.0.*`)
- [ ] Group MACs/FLOPs recursively by namespace
- [ ] Group Memory recursively by namespace
- [ ] Group Parameters recursively by namespace
- [ ] Export hierarchical JSON profile based on namespaces
- [ ] Collapse namespaces graphically in console output
- [ ] Recognize standard PyTorch export names natively (`aten::conv2d`)
- [ ] Recognize standard TensorFlow export names natively
- [ ] Handle unrecognized namespaces by clustering connected components
- [ ] Provide API to explicitly define grouping tags manually
- [ ] Analyze layer-by-layer sparsity
- [ ] Profile sequence expansion boundaries globally
- [ ] Map profiled stats directly back to Python PyTorch `nn.Module` names if metadata exists
- [ ] Highlight highly repetitive sub-structures (e.g., 24 identical transformer layers)
- [ ] Summarize average metrics per transformer block
- [ ] Summarize total attention head parameters vs feed-forward parameters
- [ ] Analyze CNN depthwise vs pointwise compute distribution
- [ ] Provide text-based terminal UI (TUI) hierarchical folding tables
- [ ] Emit CSV files mapping Node -> Layer -> Stats
- [ ] Emit Pandas DataFrame compatible dictionaries internally

### 6. Zero-Dependency & Lightweight Runtime Integrations (30+ items)
- [ ] Run profiling logic purely via `onnx9000` Python API (no native binaries)
- [ ] Support profiling >10GB LLMs directly via memory-mapped IO (without OOM)
- [ ] Execute completely within Pyodide/WASM limits
- [ ] Integrate with `Netron` visualizer as the backend profiling engine
- [ ] Expose WebWorker compatible async profiling functions
- [ ] Run instant architecture analysis in serverless functions (AWS Lambda)
- [ ] Provide CLI utility: `onnx9000 profile model.onnx`
- [ ] Output results in rich Markdown
- [ ] Output results in rich JSON
- [ ] Output results in strict CSV
- [ ] Print beautiful ASCII tables (Rich/Colorama styled) dynamically
- [ ] Track profiling execution time itself (should be sub-100ms for massive models)
- [ ] Handle disconnected graphs natively
- [ ] Generate HTML report templates
- [ ] Seamless integration directly into the `onnx9000` Graph Surgeon API (`graph.profile()`)
- [ ] Deployable instantly via CDN (JS transpiled profile logic)
- [ ] Allow streaming of results as graph is being analyzed (for extremely large models)

### 7. Extensive Profiling Edge Cases & Validations (30+ items)
- [ ] Unit Test: Profile MACs on standard ResNet50
- [ ] Unit Test: Profile FLOPs on standard ResNet50
- [ ] Unit Test: Profile Mem on standard ResNet50
- [ ] Unit Test: Profile standard BERT (Attention mechanism scaling)
- [ ] Unit Test: Profile MobileNet (Depthwise convolutions specifically)
- [ ] Unit Test: Profile Dynamic sequence lengths on Llama 3
- [ ] Unit Test: Ensure `Slice` operations handle symbolic dimensions cleanly
- [ ] Unit Test: Validate output equivalence with official `onnx-tool` counts (atol=1%)
- [ ] Unit Test: Validate peak memory allocation simulations against PyTorch native traces
- [ ] Validate `GatherND` memory access counts
- [ ] Validate `ScatterND` memory access counts
- [ ] Prevent recursion limits on extremely deep models (e.g., 1000 layers)
- [ ] Optimize symbolic resolution equations (algebraic simplification internally)
- [ ] Check `Reshape` product validity (e.g. `1 * 3 * 224 * 224 == batch * x * y * z`)
- [ ] Profile integer multiplications differently from floating point natively (if requested)
- [ ] Profile specific ONNX quantization operators (QLinearConv MACs)
- [ ] Profile dequantize -> math -> quantize cycles efficiently
- [ ] Check symbolic shape stability (variables failing to resolve completely)
- [ ] Prevent division by zero mathematically during FLOP division equations
- [ ] Validate execution against opset versions 1-21 structurally
- [ ] Fallback gracefully when encountering mathematically undefined subgraphs (e.g., RNG nodes)

### 8. External Data & Advanced Deployment Profiling (50+ items)
- [ ] Accurately profile ONNX `.bin` external data sizes natively without parsing
- [ ] Detect broken external data links during profiling
- [ ] Expose HTTP byte-range overhead analysis for streamed models
- [ ] Estimate load time given generic Network Bandwidth speeds (e.g., 10MB/s)
- [ ] Simulate chunked loading memory footprints
- [ ] Profile WebGL uniform buffer limit conflicts natively
- [ ] Profile WebGL texture size limit conflicts natively
- [ ] Profile WebGPU storage buffer alignment mismatches natively
- [ ] Analyze matrix row-major vs column-major transpose penalties explicitly
- [ ] Provide specific WASM SIMD128 memory alignment checks
- [ ] Warn on Float64 usage specifically targeting WASM limits
- [ ] Warn on dynamic memory allocations targeting WASM limits
- [ ] Flag operators that force synchronous CPU fallbacks natively
- [ ] Validate model fits securely inside Pyodide 2GB RAM limits natively
- [ ] Analyze specific graph topologies known to cause JS GC pressure (excessive small objects)
- [ ] Analyze memory-arena pre-allocation sizes specifically for JS TypedArrays

### 9. Operator-Specific FLOP/MAC Definitions (40+ items)
- [ ] Define precise FLOPs for `Einsum` natively based on equation strings
- [ ] Define precise FLOPs for `ConvInteger`
- [ ] Define precise FLOPs for `MatMulInteger`
- [ ] Define precise FLOPs for `QLinearConv`
- [ ] Define precise FLOPs for `QLinearMatMul`
- [ ] Define precise FLOPs for `LSTM` (per step and unrolled)
- [ ] Define precise FLOPs for `GRU` (per step and unrolled)
- [ ] Define precise FLOPs for `RNN` (per step and unrolled)
- [ ] Define precise FLOPs for `Multinomial`
- [ ] Define precise FLOPs for `RandomNormal`
- [ ] Define precise FLOPs for `RandomUniform`
- [ ] Define precise FLOPs for `GridSample`
- [ ] Define precise FLOPs for `RoiAlign`
- [ ] Define precise FLOPs for `MaxRoiPool`
- [ ] Define precise FLOPs for `Resize` (Nearest)
- [ ] Define precise FLOPs for `Resize` (Bilinear/Linear)
- [ ] Define precise FLOPs for `Resize` (Cubic)
- [ ] Define precise FLOPs for `SpaceToDepth` / `DepthToSpace` (Zero FLOPs, memory bound)
- [ ] Define precise FLOPs for `SpaceToDepth` memory bandwidth
- [ ] Define precise FLOPs for `DepthToSpace` memory bandwidth
- [ ] Define precise FLOPs for `Pad`
- [ ] Define precise FLOPs for `Hardmax`
- [ ] Define precise FLOPs for `LogSoftmax`
- [ ] Define precise FLOPs for `HardSigmoid`
- [ ] Define precise FLOPs for `HardSwish`
- [ ] Define precise FLOPs for `Shrink`
- [ ] Define precise FLOPs for `PRelu`
- [ ] Define precise FLOPs for `CumSum`
- [ ] Define precise FLOPs for `ReverseSequence`
- [ ] Define precise FLOPs for `BitShift`
- [ ] Define precise FLOPs for `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot`
- [ ] Define precise FLOPs for `Round`
- [ ] Define precise FLOPs for `IsInf`, `IsNaN`
- [ ] Define precise FLOPs for `SequenceConstruct`, `SequenceAt`, `SequenceEmpty`, etc.
- [ ] Differentiate memory bandwidth for `Concat` (copying) vs `Split` (viewing if supported)

### 10. Advanced Dynamic Range & Data Type Analysis (20+ items)
- [ ] Profile min/max value bounds for `Constant` tensors natively
- [ ] Profile sparsity percentage (zeros) for `Constant` tensors natively
- [ ] Warn if `Float32` constants are strictly integers visually
- [ ] Warn if `Float32` constants fit safely within `Float16` bounds without underflow
- [ ] Warn if `Int64` constants fit safely within `Int32` or `Int8` bounds
- [ ] Profile string lengths explicitly for `String` tensors
- [ ] Evaluate `BFloat16` distribution ranges specifically
- [ ] Highlight subnormal (denormal) values in `Float32` constants (can cause severe performance drops)
- [ ] Highlight NaNs or Infs explicitly baked into constants
- [ ] Analyze dynamic value ranges of `Shape` / `Size` constants if evaluated
- [ ] Provide distribution histograms of weight parameters internally (Text UI output)
- [ ] Quantify theoretical memory savings if entire graph is cast to `Float16`
- [ ] Quantify theoretical memory savings if entire graph is cast to `Int8`
- [ ] Quantify theoretical memory savings if sparsity is leveraged natively

### 11. Custom Memory Planning & Arena Simulation (20+ items)
- [ ] Simulate First-Fit contiguous memory allocation scheme
- [ ] Simulate Best-Fit contiguous memory allocation scheme
- [ ] Simulate explicit buffer reuse lifetimes based on topological traversal
- [ ] Provide explicit buffer offsets for every tensor in the simulated arena
- [ ] Support generating C/C++ header arrays describing the static memory plan
- [ ] Simulate in-place operation memory optimization natively (e.g. `Relu` modifying input)
- [ ] Simulate shared tensor memory for `Reshape` / `Flatten` / `Squeeze` (Zero-copy views)
- [ ] Calculate Peak Arena fragmentation natively
- [ ] Simulate multi-arena planning (e.g. separate arena for weights vs activations)
- [ ] Export memory plan directly to `GraphSurgeon` attributes for compiled backends

### 12. Command Line & Developer Ergonomics (10+ items)
- [ ] Output interactive HTML Flamegraphs for memory/compute profiles
- [ ] Generate D3.js TreeMaps representing hierarchical parameter distribution
- [ ] Expose native Python decorators `@onnx9000.profile` for instant model logging
- [ ] Support diffing two ONNX models natively (`onnx9000 profile A.onnx B.onnx --diff`)
- [ ] Export diff report showing exact changes in FLOPs/Params
- [ ] Implement colorized terminal outputs using standard ANSI escapes
- [ ] Graceful fallback for massive models on low-RAM machines (chunked profiling)


### 13. Advanced Hardware & Web Profiling Targets (25+ items)
- [ ] Simulate Apple Metal specific thread-group memory alignment
- [ ] Simulate WebGPU specific Workgroup alignment (e.g. multiples of 64 or 256)
- [ ] Simulate WebGL specific texture packing limits (e.g. RGBA packing overhead)
- [ ] Profile padding limits for `Conv2d` implicitly required by certain ML libraries
- [ ] Account for padding required by `int4` block unpacking logic in memory sizes
- [ ] Analyze execution overhead of implicit dimension broadcasts (broadcasting `[1,3,1,1]` to `[N,3,H,W]`)
- [ ] Analyze FLOP overhead of `Expand` when explicitly applied
- [ ] Detect implicit padding penalties for `MatMul` blocks (e.g. `K=1023` -> `1024` on TensorCores)
- [ ] Profile explicit memory usage of attention masks inside `MultiHeadAttention`
- [ ] Test symbolic inference engine limits natively via stress testing
- [ ] Unit Test: Profile `ResNet` family with explicit memory-reuse
- [ ] Unit Test: Profile `VGG` family without memory-reuse
- [ ] Unit Test: Profile `YOLO` family (dynamic bounds tracking)
- [ ] Unit Test: Profile `MobileNet` family (depthwise separation math)
- [ ] Unit Test: Profile `Transformer` (GPT-2 style dynamic sequence limits)
- [ ] Unit Test: Verify `Einsum` equations are parsed perfectly for FLOP extraction
