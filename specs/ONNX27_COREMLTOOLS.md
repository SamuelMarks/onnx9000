# ONNX27: coremltools (Web-Native Apple Silicon Bridge)

## Original Project Description
Apple's `coremltools` is a Python package that converts machine learning models from major frameworks (PyTorch, TensorFlow, ONNX) into the Core ML format (`.mlmodel` and the newer `.mlpackage`). This conversion is essential for deploying models natively on Apple hardware (macOS, iOS, iPadOS, watchOS), allowing the operating system to optimally route computations across the CPU, GPU, and the highly efficient Apple Neural Engine (ANE). The tool relies heavily on Python, protobuf generation, and underlying native binaries to optimize the graph intermediate representation (Model Intermediate Language, or MIL).

## How `onnx9000` Deviates (The WASM-First Monolith Approach)
Instead of requiring a local macOS machine with a massive Python environment to generate iOS/macOS payloads, `onnx9000.coreml` is entirely implemented in TypeScript and WebAssembly.
*   **Browser-Based Generation:** Allows developers to drag-and-drop an ONNX file into a web browser, perform MIL optimizations, and instantly download a signed `.mlpackage` ready for Xcode, entirely client-side.
*   **No Native Dependencies:** Bypasses Apple's proprietary C++ parsing libraries by implementing the CoreML protobuf schema and MIL AST directly in TypeScript.
*   **Bi-directional (CoreML -> ONNX):** While Apple's tool primarily focuses on *exporting* to CoreML, `onnx9000` natively supports *importing* existing `.mlmodel` files and lifting them back up to standard ONNX IR to run on non-Apple web environments.
*   **WebNN Integration:** Automatically generates WebNN execution hints tailored for Safari's CoreML-backed WebNN implementation, ensuring the generated graphs hit the ANE fast-paths dynamically in the browser.

---

## Exhaustive Implementation Checklist

### Phase 1: CoreML Schema & Serialization (Web-Native)
- [ ] 001. Implement CoreML protobuf parser in pure TypeScript/WASM.
- [ ] 002. Implement CoreML protobuf emitter in pure TypeScript/WASM.
- [ ] 003. Support `Model` spec version 1 to 7 compatibility flags.
- [ ] 004. Define `NeuralNetwork` schema structure.
- [ ] 005. Define `NeuralNetworkBuilder` schema structure.
- [ ] 006. Define `MILSpec.Program` schema structure (CoreML v4+).
- [ ] 007. Define `MILSpec.Function` schema structure.
- [ ] 008. Define `MILSpec.Block` schema structure.
- [ ] 009. Implement JSON-to-Protobuf serialization for CoreML definitions.
- [ ] 010. Implement `.mlmodel` flat file generator.
- [ ] 011. Implement `.mlpackage` directory structure generator.
- [ ] 012. Utilize JSZip (or WASM equivalent) to package the `.mlpackage` in the browser.
- [ ] 013. Generate `Manifest.json` for `.mlpackage`.
- [ ] 014. Generate `FeatureDescriptions.json` for `.mlpackage`.
- [ ] 015. Implement `weights/weight.bin` external data writer.
- [ ] 016. Handle chunked writing for `weight.bin` exceeding browser memory limits.
- [ ] 017. Define `FeatureType` representations (Int64, Double, String, Image, MultiArray).
- [ ] 018. Implement `ImageFeatureType` mapping (RGB, BGR, Grayscale).
- [ ] 019. Implement `DictionaryFeatureType` mapping.
- [ ] 020. Implement `SequenceFeatureType` mapping.
- [ ] 021. Parse and preserve model metadata (author, license, description).
- [ ] 022. Serialize user-defined metadata dictionaries.
- [ ] 023. Implement a linter validating the generated CoreML protobuf against Apple's strict schema.
- [ ] 024. Expose an API to read metadata from an existing `.mlmodel` without parsing weights.
- [ ] 025. Support updating model metadata natively in the browser and re-exporting.

### Phase 2: Model Intermediate Language (MIL) AST
- [ ] 026. Define base `mil.Operation` AST node.
- [ ] 027. Define base `mil.Value` and `mil.Var` nodes.
- [ ] 028. Implement MIL type system (`mil.type.tensor`, `mil.type.scalar`, `mil.type.tuple`).
- [ ] 029. Implement `mil.type.fp16`, `mil.type.fp32`, `mil.type.int32`, `mil.type.bool`.
- [ ] 030. Implement `mil.Builder` class for programmatic MIL graph construction.
- [ ] 031. Implement `mil.Program` and `mil.Function` containers.
- [ ] 032. Implement graph topological sort utility for MIL operations.
- [ ] 033. Implement MIL constant folding optimization pass.
- [ ] 034. Implement MIL dead code elimination (DCE) pass.
- [ ] 035. Implement MIL common subexpression elimination (CSE).
- [ ] 036. Implement namespace isolation for MIL variables to prevent collision.
- [ ] 037. Create the ONNX-to-MIL high-level translation loop.
- [ ] 038. Map ONNX input tensors to MIL function inputs.
- [ ] 039. Map ONNX initializers to MIL `const` operations.
- [ ] 040. Map ONNX output tensors to MIL function outputs.
- [ ] 041. Implement shape inference within the MIL AST to resolve dynamic boundaries.
- [ ] 042. Translate ONNX dynamic axes to MIL symbolic variables (e.g., `isize1`).
- [ ] 043. Handle type casting implicitly if ONNX types (e.g., int64) aren't supported natively by MIL.
- [ ] 044. Implement a textual printer for MIL AST debugging (similar to Apple's PyMIL).
- [ ] 045. Implement AST node replacement utilities for graph rewriting.
- [ ] 046. Validate MIL AST prior to lowering to Protobuf.
- [ ] 047. Track original ONNX node names in MIL metadata for debugging traceability.
- [ ] 048. Implement loop/conditional block unwrapping into standard MIL execution flow.
- [ ] 049. Establish a specific intermediate dialect `onnx9000.apple_ane` for Neural Engine specifics.
- [ ] 050. Implement a topological verifier checking against acyclic directed graph rules.

### Phase 3: Unary & Binary Arithmetic Translation (ONNX -> MIL)
- [ ] 051. Map ONNX `Add` to MIL `add`.
- [ ] 052. Map ONNX `Sub` to MIL `sub`.
- [ ] 053. Map ONNX `Mul` to MIL `mul`.
- [ ] 054. Map ONNX `Div` to MIL `real_div` / `floor_div`.
- [ ] 055. Map ONNX `Pow` to MIL `pow`.
- [ ] 056. Map ONNX `Abs` to MIL `abs`.
- [ ] 057. Map ONNX `Ceil` to MIL `ceil`.
- [ ] 058. Map ONNX `Floor` to MIL `floor`.
- [ ] 059. Map ONNX `Round` to MIL `round`.
- [ ] 060. Map ONNX `Exp` to MIL `exp`.
- [ ] 061. Map ONNX `Log` to MIL `log`.
- [ ] 062. Map ONNX `Sqrt` to MIL `sqrt`.
- [ ] 063. Map ONNX `Sin` to MIL `sin`.
- [ ] 064. Map ONNX `Cos` to MIL `cos`.
- [ ] 065. Map ONNX `Tan` to MIL `tan`.
- [ ] 066. Map ONNX `Asin` to MIL `asin`.
- [ ] 067. Map ONNX `Acos` to MIL `acos`.
- [ ] 068. Map ONNX `Atan` to MIL `atan`.
- [ ] 069. Map ONNX `Sign` to MIL `sign`.
- [ ] 070. Map ONNX `Mod` to MIL `mod`.
- [ ] 071. Map ONNX `Max` to MIL `maximum`.
- [ ] 072. Map ONNX `Min` to MIL `minimum`.
- [ ] 073. Map ONNX `Erf` to MIL `erf`.
- [ ] 074. Map ONNX `IsNaN` to MIL `isnan`.
- [ ] 075. Handle ONNX implicit broadcasting logic mapping to MIL explicit broadcast ops if needed.

### Phase 4: Tensor Manipulation Translation (ONNX -> MIL)
- [ ] 076. Map ONNX `Reshape` to MIL `reshape`.
- [ ] 077. Map ONNX `Transpose` to MIL `transpose`.
- [ ] 078. Map ONNX `Concat` to MIL `concat`.
- [ ] 079. Map ONNX `Slice` to MIL `slice_by_index` or `slice_by_size`.
- [ ] 080. Handle dynamic ONNX slice parameters using MIL symbolic bounds.
- [ ] 081. Map ONNX `Split` to MIL `split`.
- [ ] 082. Map ONNX `Squeeze` to MIL `squeeze`.
- [ ] 083. Map ONNX `Unsqueeze` to MIL `expand_dims`.
- [ ] 084. Map ONNX `Gather` to MIL `gather`.
- [ ] 085. Map ONNX `GatherElements` to MIL `gather_along_axis`.
- [ ] 086. Map ONNX `GatherND` to MIL `gather_nd`.
- [ ] 087. Map ONNX `Scatter` to MIL `scatter`.
- [ ] 088. Map ONNX `ScatterElements` to MIL `scatter_along_axis`.
- [ ] 089. Map ONNX `ScatterND` to MIL `scatter_nd`.
- [ ] 090. Map ONNX `Tile` to MIL `tile`.
- [ ] 091. Map ONNX `Pad` to MIL `pad`.
- [ ] 092. Handle `constant` padding mode in MIL.
- [ ] 093. Handle `reflect` padding mode in MIL.
- [ ] 094. Handle `edge` padding mode in MIL.
- [ ] 095. Map ONNX `Expand` to MIL `broadcast_to`.
- [ ] 096. Map ONNX `Shape` to MIL `shape`.
- [ ] 097. Map ONNX `Size` to MIL `size`.
- [ ] 098. Map ONNX `Cast` to MIL `cast`.
- [ ] 099. Identify unsupported ONNX type casting (e.g., float64) and insert warning traces.
- [ ] 100. Resolve negative axes indexing natively within the MIL translation.

### Phase 5: Neural Network Layers Translation (ONNX -> MIL)
- [ ] 101. Map ONNX `Conv` (1D/2D/3D) to MIL `conv`.
- [ ] 102. Handle convolution `dilations` translation.
- [ ] 103. Handle convolution `strides` translation.
- [ ] 104. Handle depthwise convolution via `groups` parameter.
- [ ] 105. Handle `auto_pad` string matching in MIL.
- [ ] 106. Map ONNX `ConvTranspose` to MIL `conv_transpose`.
- [ ] 107. Map ONNX `MaxPool` to MIL `max_pool`.
- [ ] 108. Map ONNX `AveragePool` to MIL `avg_pool`.
- [ ] 109. Map ONNX `GlobalMaxPool` to MIL `global_max_pool` (or pool with full spatial kernel).
- [ ] 110. Map ONNX `GlobalAveragePool` to MIL `global_avg_pool` (or pool with full spatial kernel).
- [ ] 111. Map ONNX `BatchNormalization` to MIL `batch_norm`.
- [ ] 112. Map ONNX `InstanceNormalization` to MIL `instance_norm`.
- [ ] 113. Map ONNX `LayerNormalization` to MIL `layer_norm`.
- [ ] 114. Parse and apply epsilon values correctly across all norm layers.
- [ ] 115. Map ONNX `Dropout` to a MIL Identity (since CoreML export is inference-only).
- [ ] 116. Map ONNX `MatMul` to MIL `matmul`.
- [ ] 117. Map ONNX `Gemm` to MIL `linear` (fusing alpha/beta/bias directly).
- [ ] 118. Implement padding conversions (ONNX specific spatial pads to MIL format).
- [ ] 119. Handle asymmetric padding safely within MIL convolution parameters.
- [ ] 120. Emulate ONNX `LocalResponseNormalization` (LRN) if native MIL op varies.
- [ ] 121. Emulate ONNX `MaxUnpool` utilizing indices from previous MaxPool operations.
- [ ] 122. Emulate `DepthToSpace` via MIL `pixel_shuffle`.
- [ ] 123. Emulate `SpaceToDepth` via MIL reshape and transpose sequences.
- [ ] 124. Map ONNX `Resize` to MIL `resize_bilinear` or `resize_nearest_neighbor`.
- [ ] 125. Parse coordinate transformation modes (`align_corners`, `half_pixel`) for `Resize`.

### Phase 6: Activations & Reduction Ops Translation (ONNX -> MIL)
- [ ] 126. Map ONNX `Relu` to MIL `relu`.
- [ ] 127. Map ONNX `LeakyRelu` to MIL `leaky_relu`.
- [ ] 128. Map ONNX `Sigmoid` to MIL `sigmoid`.
- [ ] 129. Map ONNX `Tanh` to MIL `tanh`.
- [ ] 130. Map ONNX `Softmax` to MIL `softmax`.
- [ ] 131. Map ONNX `LogSoftmax` to MIL `log_softmax`.
- [ ] 132. Map ONNX `Elu` to MIL `elu`.
- [ ] 133. Map ONNX `HardSigmoid` to MIL `hard_sigmoid`.
- [ ] 134. Map ONNX `Softplus` to MIL `softplus`.
- [ ] 135. Map ONNX `Softsign` to MIL `softsign`.
- [ ] 136. Map ONNX `PRelu` to MIL `prelu`.
- [ ] 137. Map ONNX `Gelu` to MIL `gelu`.
- [ ] 138. Map ONNX `Clip` to MIL `clip`.
- [ ] 139. Map ONNX `ReduceMean` to MIL `reduce_mean`.
- [ ] 140. Map ONNX `ReduceSum` to MIL `reduce_sum`.
- [ ] 141. Map ONNX `ReduceMax` to MIL `reduce_max`.
- [ ] 142. Map ONNX `ReduceMin` to MIL `reduce_min`.
- [ ] 143. Map ONNX `ReduceProd` to MIL `reduce_prod`.
- [ ] 144. Map ONNX `ReduceLogSumExp` to MIL `reduce_log_sum_exp`.
- [ ] 145. Map ONNX `ArgMax` to MIL `argmax`.
- [ ] 146. Map ONNX `ArgMin` to MIL `argmin`.
- [ ] 147. Map ONNX `NonMaxSuppression` (NMS) to MIL NMS implementations.
- [ ] 148. Map ONNX `TopK` to MIL `topk`.
- [ ] 149. Map ONNX `NonZero` to MIL `non_zero`.
- [ ] 150. Handle default `keepdims` behaviors between ONNX and MIL properly.

### Phase 7: Control Flow, Logicals, and RNNs (ONNX -> MIL)
- [ ] 151. Map ONNX `Equal` to MIL `equal`.
- [ ] 152. Map ONNX `Greater` to MIL `greater`.
- [ ] 153. Map ONNX `GreaterOrEqual` to MIL `greater_equal`.
- [ ] 154. Map ONNX `Less` to MIL `less`.
- [ ] 155. Map ONNX `LessOrEqual` to MIL `less_equal`.
- [ ] 156. Map ONNX `Not` to MIL `logical_not`.
- [ ] 157. Map ONNX `And` to MIL `logical_and`.
- [ ] 158. Map ONNX `Or` to MIL `logical_or`.
- [ ] 159. Map ONNX `Xor` to MIL `logical_xor`.
- [ ] 160. Map ONNX `Where` to MIL `select`.
- [ ] 161. Map ONNX `If` to MIL `cond`.
- [ ] 162. Map ONNX `Loop` to MIL `while_loop`.
- [ ] 163. Map ONNX `LSTM` to MIL `lstm`.
- [ ] 164. Parse LSTM direction (forward, reverse, bidirectional).
- [ ] 165. Implement state tracking for LSTM hidden variables.
- [ ] 166. Map ONNX `GRU` to MIL `gru`.
- [ ] 167. Parse GRU sequence layouts cleanly.
- [ ] 168. Map ONNX `RNN` to MIL `rnn`.
- [ ] 169. Support extraction of multiple outputs from RNN layers.
- [ ] 170. Handle static unrolling of loops if MIL dynamic control flow is unsupported in earlier spec versions.
- [ ] 171. Provide warning traces for control flow conversion potentially impacting ANE performance.
- [ ] 172. Implement `is_inf` and `is_nan` boolean mapping.
- [ ] 173. Handle ONNX `Scan` operation by unrolling it dynamically into the MIL AST.
- [ ] 174. Manage scope variables properly across MIL block boundaries.
- [ ] 175. Verify acyclic flow after parsing nested subgraphs from ONNX `If`/`Loop`.

### Phase 8: Apple Neural Engine (ANE) Specific Optimizations
- [ ] 176. Identify and rewrite MatMul sequences into 1x1 Convolutions for ANE acceleration.
- [ ] 177. Pad hidden dimensions to multiples of 64 or 32 specifically to satisfy ANE lane requirements.
- [ ] 178. Split massive convolutions (e.g., > 16384 channels) into smaller concatenated blocks to avoid ANE fallback to GPU.
- [ ] 179. Fuse sequence of `Split` -> `Concat` operations out of the graph if they cancel each other out.
- [ ] 180. Fuse `Slice` operations with adjacent `Pad` operations.
- [ ] 181. Optimize out Gather operations that index into static constants (pre-computing the gather).
- [ ] 182. Replace Swish/SiLU activations with ANE-friendly approximations if requested.
- [ ] 183. Identify LayerNorms and rewrite them into `reduce_mean`, `sub`, `pow`, `add` if explicit layer_norm causes GPU fallback on older iOS devices.
- [ ] 184. Implement an explicit ANE compatibility checker pass before finalizing the MIL AST.
- [ ] 185. Rewrite `Einsum` into explicit Transpose + MatMul + Reshape chains natively.
- [ ] 186. Pre-transpose weight constants offline to match the expected format for ANE.
- [ ] 187. Ensure 5D and 6D tensors are flattened into 4D or lower, as ANE historically struggles with higher rank shapes.
- [ ] 188. Force CAST inputs to FP16 since ANE operates almost exclusively in FP16 precision.
- [ ] 189. Map standard Transformers Attention into the specific `scaled_dot_product_attention` MIL op (requires CoreML v7/iOS 17).
- [ ] 190. Eliminate redundant `Cast` (FP32 -> FP16 -> FP32) boundaries.

### Phase 9: Compression & Quantization (`coremltools.optimize`)
- [ ] 191. Implement FP16 casting pass for all weights and biases.
- [ ] 192. Implement Palettization compression (k-means clustering of weights).
- [ ] 193. Encode LUT (Look-Up Table) weights natively into the CoreML `.weight.bin` format.
- [ ] 194. Implement INT8 Weight Quantization (W8A16) natively generating CoreML `constexpr_affine_dequantize`.
- [ ] 195. Implement INT4 Weight Quantization (W4A16) specific to CoreML iOS 17 features.
- [ ] 196. Implement sparse weight compression (storing non-zero values and bitmasks).
- [ ] 197. Support block-wise quantization grouping (e.g., group_size = 32 or 128).
- [ ] 198. Allow defining a mixed-precision configuration dictionary per layer.
- [ ] 199. Map existing ONNX `QuantizeLinear`/`DequantizeLinear` pairs directly to CoreML quantized weight representations.
- [ ] 200. Execute dynamic quantization statistics gathering natively in JS if an un-quantized model needs compression.
- [ ] 201. Support Joint-Data-Algorithm (JDA) for pruning.
- [ ] 202. Generate a compression report tracking memory reduction per layer.
- [ ] 203. Export multi-bitrate weights allowing the Apple OS to select the precision dynamically.
- [ ] 204. Handle specific iOS 17 stateful KV Cache quantization mappings.
- [ ] 205. Implement decompression validation ensuring the unpacked LUT exactly matches expected logic.

### Phase 10: Input/Output Formatting & iOS Integration
- [ ] 206. Support explicitly defining inputs as `ImageType` rather than generic `MultiArray`.
- [ ] 207. Define image scaling properties (`blueBias`, `greenBias`, `redBias`, `imageScale`).
- [ ] 208. Parse ONNX Vision transforms to automatically inject image bias metadata.
- [ ] 209. Map generic integer array outputs to specific iOS `DictionaryType` for classification tasks.
- [ ] 210. Generate the standard Core ML Class Labels file based on ONNX attributes or separate text input.
- [ ] 211. Inject custom Vision Framework descriptions directly into the generated MLModel metadata.
- [ ] 212. Provide configurable outputs mapping (renaming ONNX generic names to Swift-friendly camelCase variables).
- [ ] 213. Support defining specific input sequences as `SequenceType` for CoreML RNN wrappers.
- [ ] 214. Embed custom vocabulary files inside the `.mlpackage` for internal tokenization usage.
- [ ] 215. Configure the generated package to utilize `computeUnits = .all` explicitly by default.

### Phase 11: Bi-Directional Conversion (CoreML -> ONNX)
- [ ] 216. Implement `.mlmodel` and `.mlpackage` loader/unzipper in JS.
- [ ] 217. Parse `MILSpec.Program` back into the TypeScript AST representation.
- [ ] 218. Parse Apple NeuralNetwork V1-V3 layers (legacy protobuf) into the AST representation.
- [ ] 219. Inverse Map: MIL `conv` to ONNX `Conv`.
- [ ] 220. Inverse Map: MIL `matmul` / `linear` to ONNX `MatMul` / `Gemm`.
- [ ] 221. Inverse Map: MIL `scaled_dot_product_attention` to explicit ONNX Subgraph (MatMul, Div, Softmax, MatMul).
- [ ] 222. Extract `weight.bin` packed data back into ONNX Float32 / Float16 Initializers.
- [ ] 223. Dequantize CoreML INT4/INT8 palettized weights statically during the extraction to ONNX.
- [ ] 224. Rebuild the ONNX standard inputs/outputs definitions from the CoreML `FeatureDescription`.
- [ ] 225. Handle Swift/Apple specific renaming back to standard ONNX tensor naming conventions.
- [ ] 226. Produce a standard valid `model.onnx` payload.
- [ ] 227. Create a visual diff checker comparing the original ONNX vs the Round-Trip ONNX.

### Phase 12: GenAI & Stateful Models (iOS 18 / CoreML v8 Spec Prep)
- [ ] 228. Implement the newer Core ML Stateful operations mapping (`mil.state`).
- [ ] 229. Translate ONNX Runtime GenAI KV Cache patterns directly into CoreML state variables.
- [ ] 230. Map explicit KV-cache ring buffer updates into MIL `read_state` and `write_state`.
- [ ] 231. Translate LLaMA / Mistral ONNX topologies into stateful MLPackages natively.
- [ ] 232. Support exporting models with `Stateful=True` flags.
- [ ] 233. Generate appropriate Swift/Objective-C boilerplate text for utilizing the generated stateful model.
- [ ] 234. Map Whisper architectures efficiently to CoreML using specialized ANE-friendly audio layers.
- [ ] 235. Map Stable Diffusion UNets to CoreML natively, bypassing standard `python-coremltools` pipelines.

### Phase 13: Browser CLI & Execution Environment
- [ ] 236. Add CLI command: `onnx9000 coreml export <model.onnx>`.
- [ ] 237. Add CLI command: `onnx9000 coreml import <model.mlpackage>`.
- [ ] 238. Provide Node.js API: `import { convertToCoreML } from 'onnx9000/coreml'`.
- [ ] 239. Enable streaming conversion for files larger than 2GB (bypassing V8 array limits).
- [ ] 240. Implement Web Worker distribution for MIL AST optimization passes.
- [ ] 241. Provide a UI component: "Drop ONNX -> Get CoreML".
- [ ] 242. Display progress bars parsing protobufs natively in the browser.
- [ ] 243. Provide WebNN hint generation for macOS Safari utilizing the `coreml` WebNN backend.
- [ ] 244. Create debugging logs showing exactly which layers were offloaded to ANE vs GPU vs CPU (via simulation heuristics).
- [ ] 245. Establish memory bounds checking inside the WASM exporter to prevent browser tab crashes.

### Phase 14: Quality Assurance & Parity Testing
- [ ] 246. Validate complete conversion of ResNet50 (ONNX -> CoreML).
- [ ] 247. Validate complete conversion of MobileNetV2 (ONNX -> CoreML).
- [ ] 248. Validate complete conversion of YOLOv8 (ONNX -> CoreML).
- [ ] 249. Validate complete conversion of BERT (ONNX -> CoreML).
- [ ] 250. Validate complete conversion of GPT-2 (ONNX -> CoreML).
- [ ] 251. Validate complete conversion of Whisper-Tiny (ONNX -> CoreML).
- [ ] 252. Extract test outputs from native `coremltools` Python and compare exact tensor outputs.
- [ ] 253. Measure and ensure that generated `.mlpackage` sizes are within 1% of the Python equivalent.
- [ ] 254. Run automated tests ensuring all generated files successfully load in Xcode without validation errors.
- [ ] 255. Verify output differences of Palettized exports are mathematically acceptable (Cosine Similarity > 0.99).
- [ ] 256. Automate iOS simulator execution checking for the generated packages (using external CI/CD wrappers).
- [ ] 257. Verify that image classification labels are properly surfaced in macOS Quick Look.

### Phase 15: Edge Case & Exception Handling
- [ ] 258. Handle parsing of completely unsupported ONNX ops (e.g., custom user ops) by generating clear error traces.
- [ ] 259. Fallback cleanly if an ONNX model utilizes double precision (`float64`), forcibly downcasting to `float32`.
- [ ] 260. Manage ONNX `If`/`Loop` constructs that contain operations incompatible with MIL control flow.
- [ ] 261. Detect and warn users if their dynamic axes definitions exceed Apple's supported dimension limits.
- [ ] 262. Warn users explicitly if a specific graph topology is known to trigger ANE thermal throttling.
- [ ] 263. Catch and handle Protobuf decode failures gracefully in the browser.
- [ ] 264. Ensure generated filenames for weights inside `.mlpackage` contain no illegal characters.
- [ ] 265. Strip `\0` null terminators and non-UTF8 characters from ONNX metadata before serializing to CoreML JSON.

### Phase 16: Ecosystem & Native Integration
- [ ] 266. Enable importing HuggingFace models seamlessly: `onnx9000 coreml export hf://gpt2`.
- [ ] 267. Map Hub models to CoreML instantly using cached ONNX graphs internally.
- [ ] 268. Produce an equivalent to the `coremltools` Python API natively in TypeScript.
- [ ] 269. Support exporting multiple models into a single Pipeline `.mlmodelc` natively.
- [ ] 270. Create custom Xcode playground templates alongside the exported model.
- [ ] 271. Hook into `onnx9000.optimum` to share quantization logic directly with `onnx9000.coreml`.
- [ ] 272. Build integration examples showing the `.mlpackage` running in Swift CoreML.
- [ ] 273. Provide a React Native component demonstrating inference on the generated file.

### Phase 17: Deep CoreML v6/v7 Optimization Passes
- [ ] 274. Implement MIL `constexpr` folding dynamically to resolve constants during the AST building.
- [ ] 275. Translate ONNX `Einsum` into explicit tensor ops, bypassing ANE issues with native einsum mappings.
- [ ] 276. Identify `LayerNormalization` acting on the last dimension and utilize specific CoreML accelerated primitives.
- [ ] 277. Rewrite 1D convolutions into 2D convolutions (with height=1) as ANE highly prefers 2D topologies.
- [ ] 278. Rewrite all grouped convolutions into explicit slices/concats if targeting older iOS versions via backwards compatibility flags.
- [ ] 279. Support explicit definition of the "Compute Precision" (Float16 vs Float32) for the entire package.
- [ ] 280. Extract dynamic sequence padding operations to CPU boundaries to prevent ANE pipeline stalling.

### Phase 18: Security & Sandbox Execution
- [ ] 281. Sandbox the JSZip/Archive generation to ensure no cross-site scripting attacks via malicious model metadata.
- [ ] 282. Prevent local file-system access (except via standard Browser API prompts) during export.
- [ ] 283. Verify that parsing `.mlmodel` files doesn't trigger JS prototype pollution.
- [ ] 284. Handle extremely large dimensional definitions (e.g., trying to allocate 100GB tensors) safely without crashing the V8 engine.
- [ ] 285. Utilize Subresource Integrity (SRI) on all remote script loading inside the exported HTML demos.

### Phase 19: Documentation & Profiling
- [ ] 286. Provide comprehensive API documentation mapping `python-coremltools` functions to their TS equivalents.
- [ ] 287. Publish a migration guide: "Moving from CoreMLTools to `onnx9000`".
- [ ] 288. Generate a summary table during export showing exactly which layers are structurally modified.
- [ ] 289. Develop a mock "Profiler" simulating ANE vs GPU time based on layer topologies in the browser.
- [ ] 290. Provide visual diffing of the ONNX graph vs the generated MIL graph inside the web UI.

### Phase 20: Final Polish and Release Readiness
- [ ] 291. Implement dynamic graph batching (converting a single-batch ONNX to a multi-batch CoreML package).
- [ ] 292. Add support for specialized Apple Vision Pro (visionOS) deployment targets inside the metadata.
- [ ] 293. Build a WASM fallback for reading/writing Apple's compiled `.mlmodelc` binary directories directly.
- [ ] 294. Ensure full deterministic compilation (same ONNX + same settings = exactly identical byte output for `.mlpackage`).
- [ ] 295. Add strict linter enforcing that no proprietary/undocumented Apple MIL opcodes are generated unless explicitly requested.
- [ ] 296. Resolve all Typescript strict-mode typing errors inside the CoreML generation logic.
- [ ] 297. Configure automated CI checks running against Xcode command-line tools `coremlcompiler`.
- [ ] 298. Establish telemetry for recording which ONNX operators fail to translate most frequently.
- [ ] 299. Write comprehensive tutorial: "Bringing ONNX LLMs to iOS using `onnx9000`".
- [ ] 300. Release v1.0 complete feature parity certification matching official Apple `coremltools`.
