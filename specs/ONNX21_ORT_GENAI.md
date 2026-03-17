# ONNX19: ONNX Runtime GenAI (WASM-First Generative Execution)

## Original Project Description

ONNX Runtime GenAI (ORT GenAI) is a highly specialized extension of the standard ONNX Runtime designed specifically for executing large generative AI models (like LLMs, Whisper, and Stable Diffusion). Standard graph execution is insufficient for generative models because they require complex control loops (autoregressive decoding), dynamic memory management across sequence generations (KV Caching), and specific search algorithms (Beam Search, Top-K/Top-P sampling). ORT GenAI provides a native C++ API and custom operations to handle these generative loops efficiently, avoiding the overhead of shuttling tokens back and forth between the host language (Python) and the execution runtime.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

Instead of a separate C++ library bridging into Python, `onnx9000`'s GenAI implementation is built natively as an extension module within the web-first monolith.

- **WebAssembly / WebGPU Native Loops:** The autoregressive generation loop is implemented directly in TypeScript/WASM, preventing the catastrophic latency of crossing the JS-to-WASM boundary for every single generated token.
- **Integrated KV Cache Management:** Memory management for Key-Value caches is handled by the `onnx9000` memory planner directly in the WebGPU VRAM or WASM linear memory, utilizing ring buffers to handle infinite context window generation.
- **Unified Pipeline:** Eliminates the need for a secondary library. `onnx9000.genai` is a first-class citizen, wrapping the core execution engine to provide high-level APIs like `model.generate()`.
- **Browser-Side Tokenization:** Incorporates lightweight WASM tokenizers directly into the generation pipeline, allowing the browser to accept raw text and output raw text without server-side preprocessing.

---

## Exhaustive Implementation Checklist

### Phase 1: Core GenAI Pipeline & State Management

- [ ] 1. Define `GeneratorParams` configuration object.
- [ ] 2. Define `ModelParams` configuration object.
- [ ] 3. Implement base `Model` class for GenAI wrappers.
- [ ] 4. Implement base `Generator` class for stateful decoding.
- [ ] 5. Implement `State` object to hold execution graph and KV cache.
- [ ] 6. Create `Tensor` utility extensions specific to sequence lengths.
- [ ] 7. Implement dynamic shape allocation strategies for growing sequences.
- [ ] 8. Implement a pre-fill phase executor (processing the initial prompt).
- [ ] 9. Implement a decode phase executor (generating token by token).
- [ ] 10. Support seamless transition between pre-fill and decode phases.
- [ ] 11. Implement cross-layer KV cache synchronization.
- [ ] 12. Support for continuous batching (adding requests mid-generation).
- [ ] 13. Support for sequence batching (multiple independent sequences).
- [ ] 14. Implement paged attention memory management (conceptually in WASM/WGSL).
- [ ] 15. Handle context window overflow (sliding window cache ejection).
- [ ] 16. Implement asynchronous generation stepping (`await generator.compute_logits()`).
- [ ] 17. Implement synchronous generation stepping for blocking environments.
- [ ] 18. Support early stopping conditions (EOS token reached).
- [ ] 19. Support max length stopping conditions.
- [ ] 20. Implement a unified `generate()` high-level API.
- [ ] 21. Provide callback hooks for streaming output (yielding tokens).
- [ ] 22. Implement memory reuse across generation requests.
- [ ] 23. Handle dynamic model loading (loading weights asynchronously during pre-fill).
- [ ] 24. Implement graceful abort/cancellation of a generation loop.
- [ ] 25. Support sub-graph partitioning for multi-GPU/WebGPU chunking.

### Phase 2: Generation Algorithms (Search & Sampling)

- [ ] 26. Define `SearchOptions` configuration struct.
- [ ] 27. Implement Greedy Search algorithm.
- [ ] 28. Implement Beam Search algorithm.
- [ ] 29. Manage beam search state (beam scores, beam tokens, beam histories).
- [ ] 30. Implement beam search pruning and sorting.
- [ ] 31. Support `num_beams` parameter.
- [ ] 32. Support `num_return_sequences` parameter.
- [ ] 33. Implement multinomial sampling.
- [ ] 34. Implement Top-K sampling filter.
- [ ] 35. Implement Top-P (Nucleus) sampling filter.
- [ ] 36. Implement Min-P sampling filter.
- [ ] 37. Implement Temperature scaling applied to logits.
- [ ] 38. Support `repetition_penalty` filter.
- [ ] 39. Support `presence_penalty` filter.
- [ ] 40. Support `frequency_penalty` filter.
- [ ] 41. Support `length_penalty` filter (primarily for beam search).
- [ ] 42. Implement `no_repeat_ngram_size` filter.
- [ ] 43. Support forced BOS (Beginning of Sequence) token injection.
- [ ] 44. Support forced EOS (End of Sequence) token generation.
- [ ] 45. Implement custom logit bias injection (boosting/penalizing specific tokens).
- [ ] 46. Support custom bad words list (banning specific token sequences).
- [ ] 47. Support custom allowed words list (restricting vocabulary).
- [ ] 48. Implement diverse beam search (grouping beams to ensure variety).
- [ ] 49. Support typical decoding sampling.
- [ ] 50. Implement a modular logit processor pipeline.
- [ ] 51. Create a WASM-optimized logit sorting/filtering kernel (crucial for speed).
- [ ] 52. Create a WebGPU-optimized Top-K/Top-P extraction shader.
- [ ] 53. Implement random seed control for deterministic sampling.
- [ ] 54. Provide probability distributions per token in the output.
- [ ] 55. Implement contrastive search algorithm.

### Phase 3: Tokenization & Text Processing

- [ ] 56. Define base `Tokenizer` interface.
- [ ] 57. Implement `TokenizerStream` for real-time decoding.
- [ ] 58. Implement Byte-Pair Encoding (BPE) algorithm in WASM.
- [ ] 59. Implement WordPiece tokenization algorithm.
- [ ] 60. Implement Unigram tokenization algorithm.
- [ ] 61. Support loading HuggingFace `tokenizer.json` formats.
- [ ] 62. Support loading SentencePiece `.model` binaries natively.
- [ ] 63. Implement byte-level BPE pre-tokenization.
- [ ] 64. Implement basic whitespace pre-tokenization.
- [ ] 65. Implement punctuation splitting pre-tokenization.
- [ ] 66. Implement Unicode normalization (NFC, NFD, NFKC, NFKD).
- [ ] 67. Support added tokens (special tokens mapping).
- [ ] 68. Handle unknown `<unk>` token replacements.
- [ ] 69. Implement `encode()` method (text to token IDs).
- [ ] 70. Implement `decode()` method (token IDs to text).
- [ ] 71. Implement batched encoding.
- [ ] 72. Implement batched decoding.
- [ ] 73. Provide token ID to string lookup utilities.
- [ ] 74. Handle whitespace stripping/preservation rules cleanly.
- [ ] 75. Implement a robust Trie structure for fast token matching in WASM.
- [ ] 76. Handle UTF-8 decoding boundaries safely in streaming mode.
- [ ] 77. Integrate tokenizer instantiation via `Model.create_tokenizer()`.
- [ ] 78. Support specific tokenizer dialects (Llama, GPT-2, T5, Bert).
- [ ] 79. Implement chat template rendering (applying Jinja templates).
- [ ] 80. Fallback JS tokenizers if WASM module fails to load.

### Phase 4: KV Cache & Attention Architectures

- [ ] 81. Implement a generic `KVCache` management class.
- [ ] 82. Support standard Multi-Head Attention (MHA) caching.
- [ ] 83. Support Grouped-Query Attention (GQA) caching structures.
- [ ] 84. Support Multi-Query Attention (MQA) caching structures.
- [ ] 85. Implement continuous memory allocation for caches.
- [ ] 86. Implement fragmented (paged) memory allocation for caches.
- [ ] 87. Support past key/value graph inputs.
- [ ] 88. Support present key/value graph outputs.
- [ ] 89. Manage in-place KV cache updates (mutating past_key_values directly).
- [ ] 90. Implement rotary positional embeddings (RoPE) scaling.
- [ ] 91. Support dynamic RoPE calculation based on sequence length.
- [ ] 92. Support ALiBi (Attention with Linear Biases) positional encodings.
- [ ] 93. Implement cross-attention caching (for Encoder-Decoder models).
- [ ] 94. Optimize cache memory layouts (e.g., interleaving K and V).
- [ ] 95. Implement cache clearing/reset APIs.
- [ ] 96. Support offloading KV cache to CPU memory when WebGPU VRAM is full.
- [ ] 97. Implement cache quantization (storing K/V as int8 or fp8).
- [ ] 98. Support sliding window attention limits (e.g., Mistral).
- [ ] 99. Handle varying batch sizes between pre-fill and decode steps.
- [ ] 100. Implement specialized WebGPU shaders for fast KV cache concatenation.

### Phase 5: Model-Specific Optimizations & Architectures

- [ ] 101. Support **Llama** architecture variants (Llama 2, Llama 3).
- [ ] 102. Support **Mistral** architecture variants.
- [ ] 103. Support **Gemma** architecture variants.
- [ ] 104. Support **Phi** architecture variants (Phi-2, Phi-3).
- [ ] 105. Support **Qwen** architecture variants.
- [ ] 106. Support **GPT-NeoX** architectures.
- [ ] 107. Support **OPT** architectures.
- [ ] 108. Support **T5** (Encoder-Decoder) architectures.
- [ ] 109. Support **BART** architectures.
- [ ] 110. Support **Whisper** (Speech-to-Text) architectures.
- [ ] 111. Build specialized graph modifiers to detect and optimize Llama attention.
- [ ] 112. Implement fused FlashAttention-like kernels for WebGPU.
- [ ] 113. Implement fused FlashAttention-like kernels for WASM (SIMD).
- [ ] 114. Support weight-only quantization kernels (Int4/Int8 weights, FP32/FP16 compute).
- [ ] 115. Support AWQ (Activation-aware Weight Quantization) execution.
- [ ] 116. Support GPTQ execution.
- [ ] 117. Handle custom vocabulary sizes dynamically.
- [ ] 118. Implement MoE (Mixture of Experts) expert routing natively.
- [ ] 119. Handle dynamic expert loading for MoE models in the browser.
- [ ] 120. Optimize feed-forward network (FFN) fusions (SwiGLU, GeGLU).
- [ ] 121. Support multi-modal inputs (passing image embeddings to LLMs).
- [ ] 122. Implement vision encoder pipelines (e.g., CLIP) alongside GenAI.
- [ ] 123. Support LoRA (Low-Rank Adaptation) adapter loading.
- [ ] 124. Enable dynamic swapping of LoRA adapters during generation.
- [ ] 125. Support speculative decoding (using a draft model to accelerate target model).

### Phase 6: API Mappings & Web Integration

- [ ] 126. Create `onnx9000.genai.Model` Python API.
- [ ] 127. Create `onnx9000.genai.GeneratorParams` Python API.
- [ ] 128. Create `onnx9000.genai.Tokenizer` Python API.
- [ ] 129. Create TypeScript bindings: `onnx9000-genai.ts`.
- [ ] 130. Export `GeneratorParams` TS interface.
- [ ] 131. Export `Model` TS interface.
- [ ] 132. Export `Tokenizer` TS interface.
- [ ] 133. Implement a Web Worker dedicated to the GenAI execution loop.
- [ ] 134. Create a messaging protocol between main thread and GenAI worker.
- [ ] 135. Expose an `AsyncGenerator` for TS streaming output (`for await (const token of ...)`).
- [ ] 136. Support passing existing WebGPU device instances to the GenAI model.
- [ ] 137. Implement memory progress callbacks (for downloading large LLM weights).
- [ ] 138. Handle indexedDB caching of downloaded `.onnx` and `.safetensors` files.
- [ ] 139. Implement automated hardware capability detection (selecting WASM vs WebGPU).
- [ ] 140. Expose profiling data (tokens/sec, time-to-first-token) via the API.
- [ ] 141. Ensure garbage collection of generator state when streams are closed.
- [ ] 142. Support standard HuggingFace `generation_config.json` loading.
- [ ] 143. Build an OpenAI-compatible REST API wrapper utilizing `onnx9000.genai` under the hood.
- [ ] 144. Create a local web server utility serving the OpenAI-compatible endpoints.
- [ ] 145. Implement cross-origin resource sharing (CORS) configurations for local serving.

### Phase 7: Generative Builders & Export Tooling

- [ ] 146. Create `onnx9000.genai.builder` module to prepare standard models for GenAI.
- [ ] 147. Implement a PyTorch to ONNX exporter specifically tuned for GenAI graph structures.
- [ ] 148. Automate the insertion of KV cache inputs/outputs during export.
- [ ] 149. Automate the conversion of static sequence lengths to dynamic axes.
- [ ] 150. Implement a graph pass to remove unwanted past-state initializers.
- [ ] 151. Build a CLI command: `onnx9000 genai build <model_id> --target webgpu`.
- [ ] 152. Build a CLI command: `onnx9000 genai chat <model_path>`.
- [ ] 153. Implement automatic folder structuring (creating `model.onnx`, `tokenizer.json`, etc.).
- [ ] 154. Support splitting large models into chunks (`model-001.onnx`, `model-002.onnx`).
- [ ] 155. Generate a manifest file describing the model chunk layout.
- [ ] 156. Implement weight externalization during export to minimize proto size.
- [ ] 157. Validate exported model structures against GenAI requirements.
- [ ] 158. Provide an automated quantization step during the build process (e.g., INT4 block quantization).
- [ ] 159. Support exporting with embedded tokenizers (storing tokenizer config inside ONNX metadata).
- [ ] 160. Create integration scripts for downloading directly from HuggingFace Hub.

### Phase 8: Advanced Inference Techniques

- [ ] 161. Implement prompt caching (saving KV states of frequent system prompts to disk/IDB).
- [ ] 162. Implement batched prompt processing (processing prefix trees efficiently).
- [ ] 163. Support grammar-guided generation (Constrained Decoding via BNF/EBNF).
- [ ] 164. Support JSON schema-guided generation.
- [ ] 165. Implement regex-guided generation.
- [ ] 166. Handle stopping criteria based on complex string matching.
- [ ] 167. Implement lookahead decoding techniques.
- [ ] 168. Implement Medusa/EAGLE head support (generating multiple tokens per step).
- [ ] 169. Support watermarking of generated text (e.g., Kirchenbauer et al.).
- [ ] 170. Implement prefix matching for fast retrieval-augmented generation (RAG) updates.

### Phase 9: UI Components & Demos

- [ ] 171. Build a barebones HTML/JS demo demonstrating browser-local Llama execution.
- [ ] 172. Create a React hook: `useGenAI(modelUrl)`.
- [ ] 173. Create a Vue composable: `useGenAI(modelUrl)`.
- [ ] 174. Implement a terminal UI (TUI) chat interface for the CLI.
- [ ] 175. Create a WebGL/Canvas visualizer showing token probabilities in real-time.
- [ ] 176. Implement a drag-and-drop interface for loading local ONNX files.
- [ ] 177. Provide a progressive web app (PWA) wrapper for offline GenAI usage.
- [ ] 178. Create an example showing Whisper audio transcription feeding into an LLM.
- [ ] 179. Create an example demonstrating streaming JSON extraction from unstructured text.
- [ ] 180. Document best practices for memory management in mobile browsers.

### Phase 10: Performance, Testing, and Compliance

- [ ] 181. Create unit tests for all logit processors.
- [ ] 182. Create unit tests for beam search logic.
- [ ] 183. Create unit tests for KV cache indexing math.
- [ ] 184. Implement fuzzing for the BPE tokenizer logic.
- [ ] 185. Benchmark Time-To-First-Token (TTFT) against standard ONNX Runtime.
- [ ] 186. Benchmark Tokens-Per-Second (TPS) across various prompt lengths.
- [ ] 187. Profile memory consumption during max-context generation.
- [ ] 188. Ensure exact numerical parity (or within acceptable tolerance) with HuggingFace Transformers outputs.
- [ ] 189. Create a regression test suite using known prompts and expected outputs.
- [ ] 190. Verify correct handling of zero-length prompts.
- [ ] 191. Verify correct handling of prompts exceeding the maximum context window.
- [ ] 192. Ensure correct batch padding behavior in sequence batching.
- [ ] 193. Implement logging for generation statistics (prompt tokens, completion tokens, times).
- [ ] 194. Document the process for supporting a new model architecture.
- [ ] 195. Create automated memory leak detection tests for the generator loop.
- [ ] 196. Validate WebGPU shader precision constraints on different OS/Driver combinations.
- [ ] 197. Ensure WASM fallback matches WebGPU output deterministically.
- [ ] 198. Establish CI pipeline specifically for GenAI heavy integration tests.
- [ ] 199. Write comprehensive API documentation for the `onnx9000.genai` namespace.
- [ ] 200. Achieve 100% test coverage for the core generation loop state machine.

### Phase 11: Text-to-Image / Multi-modal GenAI

- [ ] 201. Define `ImageGeneratorParams`.
- [ ] 202. Implement UNet/DiT inference loop for diffusion models.
- [ ] 203. Implement VAE (Variational Autoencoder) decoding step.
- [ ] 204. Support DDIM scheduler.
- [ ] 205. Support Euler Ancestral scheduler.
- [ ] 206. Support PNDM scheduler.
- [ ] 207. Support LCM (Latent Consistency Model) schedulers.
- [ ] 208. Implement classifier-free guidance (CFG) logic.
- [ ] 209. Support negative prompts handling.
- [ ] 210. Implement latent noise generation with controlled seeds.
- [ ] 211. Manage the multi-model pipeline (Text Encoder -> UNet -> VAE).
- [ ] 212. Support Stable Diffusion v1.5 architectures.
- [ ] 213. Support Stable Diffusion XL (SDXL) architectures.
- [ ] 214. Support image-to-image generation (adding noise to base image).
- [ ] 215. Support inpainting (handling mask inputs in the UNet loop).
- [ ] 216. Implement ControlNet support alongside the UNet.
- [ ] 217. Expose progressive image generation hooks (yielding partial images).
- [ ] 218. Support exporting the VAE output directly to an HTML Canvas `ImageData` object.
- [ ] 219. Handle dynamic resolution scaling.
- [ ] 220. Implement memory optimizations for the diffusion loop to prevent WebGPU crashes.

### Phase 12: Audio GenAI

- [ ] 221. Support Text-to-Speech (TTS) architectures (e.g., VITS).
- [ ] 222. Support Bark architecture.
- [ ] 223. Support MusicGen architecture.
- [ ] 224. Implement streaming audio output (yielding PCM chunks).
- [ ] 225. Handle mel-spectrogram generation loops.
- [ ] 226. Integrate with Web Audio API for direct playback.
- [ ] 227. Implement vocoder decoding logic.
- [ ] 228. Handle multi-speaker embeddings.
- [ ] 229. Ensure continuous audio generation without clicking artifacts.
- [ ] 230. Provide Python and JS APIs for saving generated audio to `.wav`.

### Phase 13: Edge Case Handling & Stability

- [ ] 231. Handle extremely large vocabularies (>100k tokens) without memory bloat.
- [ ] 232. Manage Out-Of-Memory (OOM) WebGPU errors gracefully, falling back or shrinking cache.
- [ ] 233. Handle NaN/Inf propagation during generation (resetting or skipping).
- [ ] 234. Support aborting a generation request based on an external `AbortSignal`.
- [ ] 235. Validate inputs against expected model shapes to prevent silent failures.
- [ ] 236. Ensure thread safety in Python multi-threading environments for the generator.
- [ ] 237. Ensure Web Worker isolation and lifecycle management in the browser.
- [ ] 238. Provide robust error messages for malformed chat templates.
- [ ] 239. Handle unexpected end-of-stream during model file downloading.
- [ ] 240. Implement a safe mode that disables advanced sampling if incompatibilities arise.

### Phase 14: Ecosystem Integration

- [ ] 241. Provide integration examples with LangChain.js.
- [ ] 242. Provide integration examples with LlamaIndex.TS.
- [ ] 243. Create a unified pipeline representation for sharing GenAI models on standard model hubs.
- [ ] 244. Implement a conversion script mapping standard `GGUF` files to `onnx9000` GenAI packages.
- [ ] 245. Support consuming metadata directly from HuggingFace `config.json`.
- [ ] 246. Provide typing definitions compatible with major TS frameworks (Next.js, Nuxt).
- [ ] 247. Create a Discord/Slack bot template using the local GenAI engine.
- [ ] 248. Integrate with local vector databases for completely offline RAG applications.
- [ ] 249. Publish benchmark comparisons against standard `llama.cpp` and `onnxruntime-genai`.
- [ ] 250. Release final `v1.0` feature parity certification for GenAI capabilities.

### Phase 15: Deep WASM/WebGPU Optimizations

- [ ] 251. Implement WebGPU buffer mapping to avoid redundant CPU-GPU copies during logit retrieval.
- [ ] 252. Optimize WASM indirect function calls in the generation loop.
- [ ] 253. Utilize WebGPU `compute` subgroups for fast logit reduction (max/sum) if available.
- [ ] 254. Implement ring-buffer logic purely in WGSL to manage KV cache without host intervention.
- [ ] 255. Support asynchronous WebGPU pipeline compilation during pre-fill phase to hide latency.
- [ ] 256. Implement custom memory allocators in WASM specifically tuned for tensor lifecycle in GenAI.
- [ ] 257. Profile and minimize JS garbage collection pauses during streaming.
- [ ] 258. Support 16-bit float (fp16) WebGPU extensions globally for GenAI graphs.
- [ ] 259. Implement pre-fetching of next-layer weights in memory-constrained environments.
- [ ] 260. Develop custom shader generation strategies specifically for MoE routing logic.

### Phase 16: Extended Features

- [ ] 261. Support drafting models for speculative decoding (running a small model to predict multiple tokens).
- [ ] 262. Verify drafted tokens with the target model efficiently in a single pass.
- [ ] 263. Implement self-consistency decoding (generating multiple paths and choosing the majority).
- [ ] 264. Support explicit continuous batching API (adding sequences to an active batch queue).
- [ ] 265. Implement a priority queue for continuous batching requests.
- [ ] 266. Expose intermediate hidden states during generation for visualization or analysis.
- [ ] 267. Implement prompt compression algorithms.
- [ ] 268. Support chunked pre-filling (processing very long prompts in blocks to maintain UI responsiveness).
- [ ] 269. Allow dynamic adjustment of generation parameters (e.g., temperature) mid-generation.
- [ ] 270. Handle multi-turn conversation caching inherently within the `State` object.

### Phase 17: Multi-GPU / Distributed Execution

- [ ] 271. Implement basic tensor parallelism (splitting weights across multiple WebGPU devices/contexts).
- [ ] 272. Handle inter-device synchronization for multi-GPU setups.
- [ ] 273. Support pipeline parallelism (allocating layers sequentially to different workers/devices).
- [ ] 274. Create a master coordinator script for multi-worker distributed browser execution.
- [ ] 275. Handle node failure/dropouts in a browser-based distributed generation setup.
- [ ] 276. Implement communication primitives (AllReduce, AllGather) using WebRTC or BroadcastChannel.
- [ ] 277. Profile communication overhead vs compute gains in distributed browser environments.
- [ ] 278. Build a demonstration of collaborative inference (multiple users generating a single response).
- [ ] 279. Support distributing the KV cache across multiple devices.
- [ ] 280. Establish security protocols for sharing tensor data between browser contexts.

### Phase 18: Quality Assurance & Tooling

- [ ] 281. Build a specialized debugger UI stepping through the generation loop token by token.
- [ ] 282. Visualize attention maps generated during decoding.
- [ ] 283. Visualize beam search trees dynamically.
- [ ] 284. Implement a linter for custom sampling configurations.
- [ ] 285. Generate detailed trace logs (compatible with Chrome Tracing) for the GenAI pipeline.
- [ ] 286. Create a suite of "broken" models to ensure the runtime fails gracefully.
- [ ] 287. Maintain a known-issues database mapped to specific hardware driver bugs (e.g., specific Android WebGPU issues).
- [ ] 288. Automate testing of specific tokenizer corner cases (emoji, complex unicode).
- [ ] 289. Provide a script to compare `onnx9000` logit outputs directly with Python PyTorch tensors.
- [ ] 290. Implement a feature toggle system to disable experimental GenAI optimizations.

### Phase 19: Security & Safety

- [ ] 291. Implement prompt injection detection heuristics.
- [ ] 292. Integrate with content safety filters (e.g., Llama Guard) seamlessly in the pipeline.
- [ ] 293. Ensure secure execution boundaries for Web Workers handling third-party models.
- [ ] 294. Prevent malicious `.onnx` files from exploiting the GenAI memory allocator.
- [ ] 295. Sanitize chat templates to prevent arbitrary code execution via template injection.
- [ ] 296. Implement resource limits to prevent denial-of-service via infinite generation loops.
- [ ] 297. Support encrypted model execution (decrypting weights dynamically in WASM memory).
- [ ] 298. Validate digital signatures of downloaded GenAI packages.
- [ ] 299. Ensure no sensitive KV cache data leaks between independent generation requests.
- [ ] 300. Maintain strict Content Security Policy (CSP) compliance for web deployments.
