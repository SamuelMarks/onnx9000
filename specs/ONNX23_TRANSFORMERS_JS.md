# ONNX20: Transformers.js (WASM-Native Auto-Pipelines)

## Original Project Description
Transformers.js is a wildly popular JavaScript port of Hugging Face's `transformers` Python library. It enables developers to run pre-trained models (text, vision, audio, multimodal) directly in the browser or Node.js. It achieves this by combining `onnxruntime-web` for tensor execution with pure-JavaScript implementations of tokenizers, feature extractors, and data processors. It abstracts away the complexity of model execution by providing the `pipeline()` API, allowing users to perform tasks like sentiment analysis, image classification, or speech recognition with just a few lines of code.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)
Instead of acting as a JavaScript wrapper around a massive compiled C++ runtime (`onnxruntime-web`) and relying on slow pure-JS data processors, `onnx9000` integrates the Transformers ecosystem natively into its AOT/WASM core.
*   **WASM-Accelerated Processors:** Image resizing, Mel-spectrogram generation, and BPE tokenization are implemented as highly optimized WASM modules rather than pure JS, preventing UI thread blocking and offering near-native data preparation speeds.
*   **Zero-Overhead Inference:** Uses `onnx9000`'s lightweight runtime or AOT-compiled WebGPU shaders instead of a 2MB-5MB generic execution provider.
*   **Unified AutoClasses:** The Python and TypeScript/Browser codebases share the same architectural logic via the monolith, meaning a model supported in Python `onnx9000` is instantly available in the browser via `onnx9000.transformers`.
*   **WebGPU First:** All Vision and Audio processing tensors seamlessly share memory spaces with the execution backend (WebGPU), eliminating expensive CPU-to-GPU memory copying during pipeline execution.

---

## Exhaustive Implementation Checklist

### Phase 1: Pipeline API & Task Orchestration
- [ ] 001. Implement the base `Pipeline` class.
- [ ] 002. Implement the `pipeline(task, model, ...)` factory function.
- [ ] 003. Support `feature-extraction` pipeline (getting hidden states).
- [ ] 004. Support `text-classification` pipeline (e.g., sentiment analysis).
- [ ] 005. Support `token-classification` pipeline (e.g., NER, POS tagging).
- [ ] 006. Support `question-answering` pipeline.
- [ ] 007. Support `zero-shot-classification` pipeline.
- [ ] 008. Support `translation` pipeline.
- [ ] 009. Support `summarization` pipeline.
- [ ] 010. Support `text-generation` pipeline (integrating with ONNX19 GenAI).
- [ ] 011. Support `text2text-generation` pipeline.
- [ ] 012. Support `fill-mask` pipeline.
- [ ] 013. Support `image-classification` pipeline.
- [ ] 014. Support `object-detection` pipeline.
- [ ] 015. Support `zero-shot-image-classification` pipeline.
- [ ] 016. Support `image-segmentation` pipeline.
- [ ] 017. Support `depth-estimation` pipeline.
- [ ] 018. Support `image-to-image` pipeline.
- [ ] 019. Support `audio-classification` pipeline.
- [ ] 020. Support `automatic-speech-recognition` (ASR) pipeline.
- [ ] 021. Support `text-to-speech` (TTS) pipeline.
- [ ] 022. Support `document-question-answering` pipeline.
- [ ] 023. Support `visual-question-answering` pipeline.
- [ ] 024. Support `image-feature-extraction` pipeline.
- [ ] 025. Support pipeline batching (`[input1, input2]`).
- [ ] 026. Implement `top_k` argument parsing in classification pipelines.
- [ ] 027. Implement thresholding arguments in detection pipelines.
- [ ] 028. Support generic `device` flag (mapping to WebGPU/WASM).
- [ ] 029. Support `dtype` casting in pipelines (fp32, fp16, int8).
- [ ] 030. Implement progressive callbacks in pipelines (for streaming or download progress).
- [ ] 031. Implement pipeline pooling (keeping models hot in memory).
- [ ] 032. Allow custom pre_process step overriding in pipelines.
- [ ] 033. Allow custom post_process step overriding in pipelines.
- [ ] 034. Allow forward step overriding in pipelines.
- [ ] 035. Ensure structured error throwing for unsupported pipeline/model combos.

### Phase 2: Tokenizer Engine (Full HF Compatibility)
- [ ] 036. Define `PreTrainedTokenizer` base class.
- [ ] 037. Define `PreTrainedTokenizerFast` base class (WASM backed).
- [ ] 038. Support loading `tokenizer_config.json`.
- [ ] 039. Support loading `tokenizer.json` (the fast tokenizer format).
- [ ] 040. Implement WASM BPE (Byte-Pair Encoding) implementation.
- [ ] 041. Implement WASM WordPiece implementation.
- [ ] 042. Implement WASM Unigram implementation.
- [ ] 043. Handle `padding="max_length"` keyword argument.
- [ ] 044. Handle `padding="longest"` keyword argument.
- [ ] 045. Handle `padding=False` keyword argument.
- [ ] 046. Handle `truncation=True` keyword argument.
- [ ] 047. Handle `truncation="only_first"` keyword argument.
- [ ] 048. Handle `truncation="only_second"` keyword argument.
- [ ] 049. Handle `truncation="longest_first"` keyword argument.
- [ ] 050. Handle `max_length` keyword argument.
- [ ] 051. Handle `stride` keyword argument for overlapping contexts.
- [ ] 052. Handle `return_tensors` ("np", "pt", "tf", "ort", "webgpu").
- [ ] 053. Handle `return_attention_mask` keyword argument.
- [ ] 054. Handle `return_token_type_ids` keyword argument.
- [ ] 055. Handle `return_overflowing_tokens` keyword argument.
- [ ] 056. Handle `return_special_tokens_mask` keyword argument.
- [ ] 057. Handle `return_offsets_mapping` keyword argument.
- [ ] 058. Implement word to token ID mapping (`word_ids()`).
- [ ] 059. Implement character to token ID mapping (`char_to_token()`).
- [ ] 060. Implement token to character mapping (`token_to_chars()`).
- [ ] 061. Support text pairs (Sentence A, Sentence B).
- [ ] 062. Implement special token addition logic.
- [ ] 063. Handle `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`.
- [ ] 064. Process complex `AddedToken` configurations (lstrip, rstrip, single_word).
- [ ] 065. Implement `decode()` and `batch_decode()`.
- [ ] 066. Support `skip_special_tokens` in decoding.
- [ ] 067. Support `clean_up_tokenization_spaces` in decoding.
- [ ] 068. Implement regex-based pre-tokenizers in WASM.
- [ ] 069. Implement byte-level pre-tokenizers.
- [ ] 070. Implement Metaspace pre-tokenizers.
- [ ] 071. Implement punctuation splitting pre-tokenizers.
- [ ] 072. Implement decoders (ByteLevel, WordPiece, Metaspace).
- [ ] 073. Provide a fallback JS implementation for non-WASM environments.
- [ ] 074. Implement chat templates using a lightweight JS Jinja engine.
- [ ] 075. Validate inputs (strings, lists of strings, nested lists).

### Phase 3: Vision Processors & Image Handling
- [ ] 076. Define `BaseImageProcessor` interface.
- [ ] 077. Create `onnx9000.Image` object wrapper (handling Canvas/ImageData/Blob/URL).
- [ ] 078. Support loading images directly from URLs natively.
- [ ] 079. Support loading images from base64 strings.
- [ ] 080. Implement `do_resize` logic.
- [ ] 081. Implement WASM-accelerated bilinear interpolation resizing.
- [ ] 082. Implement WASM-accelerated bicubic interpolation resizing.
- [ ] 083. Implement WASM-accelerated nearest-neighbor interpolation resizing.
- [ ] 084. Implement `do_center_crop` logic.
- [ ] 085. Implement `do_random_crop` logic.
- [ ] 086. Implement `do_pad` logic (constant, reflect, edge padding).
- [ ] 087. Implement `do_rescale` (e.g., multiplying by 1/255).
- [ ] 088. Implement `do_normalize` (subtracting mean, dividing by std).
- [ ] 089. Support custom `image_mean` and `image_std` parameters.
- [ ] 090. Handle layout conversion (HWC to CHW format).
- [ ] 091. Implement `ImageProcessor` batching (lists of images).
- [ ] 092. Support `return_tensors` specifically for WebGPU image uploads.
- [ ] 093. Create specialized `ViTImageProcessor`.
- [ ] 094. Create specialized `CLIPImageProcessor`.
- [ ] 095. Create specialized `DeiTImageProcessor`.
- [ ] 096. Create specialized `DetrImageProcessor`.
- [ ] 097. Create specialized `YolosImageProcessor`.
- [ ] 098. Implement bounding box drawing utilities on HTML Canvas.
- [ ] 099. Implement segmentation mask drawing utilities on HTML Canvas.
- [ ] 100. Write WebGPU shaders for on-device image normalization to bypass CPU.
- [ ] 101. Write WebGPU shaders for on-device image resizing.
- [ ] 102. Support Exif orientation correction before processing.
- [ ] 103. Handle RGBA to RGB conversion.
- [ ] 104. Handle Grayscale to RGB conversion.
- [ ] 105. Optimize raw pixel array copying into WASM heap.

### Phase 4: Audio Processors (Feature Extractors)
- [ ] 106. Define `SequenceFeatureExtractor` base class.
- [ ] 107. Create `onnx9000.Audio` object wrapper.
- [ ] 108. Support loading audio from URLs.
- [ ] 109. Support loading audio from Blob/File objects.
- [ ] 110. Integrate `AudioContext` for in-browser audio decoding.
- [ ] 111. Implement WASM-accelerated 1D audio resampling.
- [ ] 112. Implement `do_pad` for audio (zero padding, reflection).
- [ ] 113. Implement `do_truncate` for audio sequence lengths.
- [ ] 114. Support `return_attention_mask` for padded audio sequences.
- [ ] 115. Implement Short-Time Fourier Transform (STFT) in WASM.
- [ ] 116. Implement Windowing functions (Hann, Hamming, Mel) in WASM.
- [ ] 117. Implement Mel-filterbank matrix generation.
- [ ] 118. Implement Mel-spectrogram computation pipeline (STFT -> Power -> Mel).
- [ ] 119. Implement log10 application for log-mel spectrograms.
- [ ] 120. Create specialized `WhisperFeatureExtractor`.
- [ ] 121. Create specialized `Wav2Vec2FeatureExtractor`.
- [ ] 122. Create specialized `SpeechT5FeatureExtractor`.
- [ ] 123. Implement raw waveform chunking (for long-form audio processing).
- [ ] 124. Handle multi-channel audio (downmixing to mono).
- [ ] 125. Normalize audio amplitude arrays (zero mean, unit variance).
- [ ] 126. Support Voice Activity Detection (VAD) pre-processing (optional extension).
- [ ] 127. Implement Web Audio API Worklet for streaming feature extraction.
- [ ] 128. Optimize memory usage during large Mel-spectrogram generation.
- [ ] 129. Implement audio output formatters (Float32Array to WAV blob).
- [ ] 130. Ensure floating point determinism across JS, WASM, and WebGPU for audio ops.

### Phase 5: Auto Classes & Hub Integration
- [ ] 131. Implement `AutoConfig.from_pretrained()`.
- [ ] 132. Implement `AutoTokenizer.from_pretrained()`.
- [ ] 133. Implement `AutoFeatureExtractor.from_pretrained()`.
- [ ] 134. Implement `AutoProcessor.from_pretrained()`.
- [ ] 135. Implement `AutoModel.from_pretrained()`.
- [ ] 136. Implement `AutoModelForSequenceClassification`.
- [ ] 137. Implement `AutoModelForTokenClassification`.
- [ ] 138. Implement `AutoModelForQuestionAnswering`.
- [ ] 139. Implement `AutoModelForCausalLM`.
- [ ] 140. Implement `AutoModelForMaskedLM`.
- [ ] 141. Implement `AutoModelForSeq2SeqLM`.
- [ ] 142. Implement `AutoModelForImageClassification`.
- [ ] 143. Implement `AutoModelForObjectDetection`.
- [ ] 144. Implement `AutoModelForSpeechSeq2Seq`.
- [ ] 145. Implement fetching from `hf.co` Hub REST API.
- [ ] 146. Support custom Hub endpoints/mirrors.
- [ ] 147. Implement API Key authentication for private models.
- [ ] 148. Support `revision` flag (fetching specific branches/commits).
- [ ] 149. Support resolving ONNX filenames (`model.onnx`, `model_quantized.onnx`).
- [ ] 150. Implement IndexedDB caching via CacheStorage API for models.
- [ ] 151. Implement ETag checking to prevent redundant model downloads.
- [ ] 152. Implement concurrent multipart file downloading for large models.
- [ ] 153. Implement fallback caching strategies for Node.js (`fs`).
- [ ] 154. Support reading models from local directory paths.
- [ ] 155. Provide an API to clear/manage the downloaded model cache.

### Phase 6: Core Model Execution Wrappers
- [ ] 156. Define `PreTrainedModel` base class.
- [ ] 157. Connect `PreTrainedModel` to the `onnx9000` execution backend.
- [ ] 158. Implement model initialization logic (loading weights into WebGPU/WASM).
- [ ] 159. Parse `config.json` to configure model input/output layers dynamically.
- [ ] 160. Manage `session_options` (threads, execution providers).
- [ ] 161. Implement the `__call__` / `forward` method abstracting the inference session.
- [ ] 162. Handle dynamic batch size resolution prior to graph execution.
- [ ] 163. Map model-specific input names (e.g., `input_ids`, `pixel_values`).
- [ ] 164. Map model-specific output names (e.g., `logits`, `last_hidden_state`).
- [ ] 165. Support external data format files (`model.onnx_data`).
- [ ] 166. Implement automatic input casting (e.g., BigInt64 to Int32 for WASM).
- [ ] 167. Handle `attention_mask` application internally if required by specific ops.
- [ ] 168. Attach the `GenerationMixin` for text-generative models.
- [ ] 169. Provide explicit memory disposal methods (`model.dispose()`).
- [ ] 170. Create debugging mode to trace input/output shapes per execution step.

### Phase 7: Post-Processing & Output Generation
- [ ] 171. Implement generic `post_process` hooks.
- [ ] 172. Implement post-processing for Text Classification (applying Softmax, indexing `id2label`).
- [ ] 173. Implement post-processing for Token Classification (aggregating sub-words, aligning offsets).
- [ ] 174. Implement post-processing for Question Answering (finding max start/end logits, span extraction).
- [ ] 175. Implement post-processing for Zero-Shot Classification (NLI entailment/contradiction mapping).
- [ ] 176. Implement post-processing for Image Classification (Softmax -> Top K).
- [ ] 177. Implement post-processing for Object Detection.
- [ ] 178. Build Non-Maximum Suppression (NMS) in WASM.
- [ ] 179. Build bounding box denormalization (cx,cy,w,h to xmin,ymin,xmax,ymax).
- [ ] 180. Implement post-processing for Semantic Segmentation (argmax over spatial dims).
- [ ] 181. Support chunked output decoding for ASR (Whisper timestamps processing).
- [ ] 182. Construct `ModelOutput` classes (similar to HF dictionaries).
- [ ] 183. Support raw output returning (`return_tensors=True` on pipelines).
- [ ] 184. Support streaming generation responses (Generators/AsyncIterators).

### Phase 8: NLP Architecture Support (Validation)
- [ ] 185. Validate end-to-end `BERT` pipeline.
- [ ] 186. Validate end-to-end `RoBERTa` pipeline.
- [ ] 187. Validate end-to-end `DistilBERT` pipeline.
- [ ] 188. Validate end-to-end `ALBERT` pipeline.
- [ ] 189. Validate end-to-end `DeBERTa` pipeline.
- [ ] 190. Validate end-to-end `MobileBERT` pipeline.
- [ ] 191. Validate end-to-end `T5` pipeline.
- [ ] 192. Validate end-to-end `BART` pipeline.
- [ ] 193. Validate end-to-end `MarianMT` pipeline.
- [ ] 194. Validate end-to-end `GPT-2` pipeline.
- [ ] 195. Validate end-to-end `LLaMA` pipeline (integrating GenAI capabilities).
- [ ] 196. Validate end-to-end `Mistral` pipeline.
- [ ] 197. Validate end-to-end `Gemma` pipeline.
- [ ] 198. Validate end-to-end `Phi` pipeline.
- [ ] 199. Handle missing token type IDs cleanly for architectures that ignore them.
- [ ] 200. Ensure position ID injection works for models without internal generators.

### Phase 9: Vision & Audio Architecture Support (Validation)
- [ ] 201. Validate end-to-end `ViT` (Vision Transformer) pipeline.
- [ ] 202. Validate end-to-end `ResNet` pipeline.
- [ ] 203. Validate end-to-end `Swin` Transformer pipeline.
- [ ] 204. Validate end-to-end `MobileNetV2` pipeline.
- [ ] 205. Validate end-to-end `ConvNeXT` pipeline.
- [ ] 206. Validate end-to-end `DETR` pipeline.
- [ ] 207. Validate end-to-end `YOLOS` pipeline.
- [ ] 208. Validate end-to-end `SegFormer` pipeline.
- [ ] 209. Validate end-to-end `CLIP` pipeline (Image + Text).
- [ ] 210. Validate end-to-end `OwlViT` pipeline.
- [ ] 211. Validate end-to-end `BLIP` pipeline.
- [ ] 212. Validate end-to-end `TrOCR` pipeline.
- [ ] 213. Validate end-to-end `Whisper` pipeline (ASR).
- [ ] 214. Validate end-to-end `Wav2Vec2` pipeline (ASR).
- [ ] 215. Validate end-to-end `SpeechT5` pipeline (TTS).
- [ ] 216. Validate end-to-end `Hubert` pipeline.
- [ ] 217. Validate end-to-end `Clap` pipeline.

### Phase 10: Utility, Math & Tensor Interop
- [ ] 218. Implement `softmax(tensor, axis)` utility.
- [ ] 219. Implement `log_softmax(tensor, axis)` utility.
- [ ] 220. Implement `sigmoid(tensor)` utility.
- [ ] 221. Implement `get_top_k(tensor, k)` utility.
- [ ] 222. Implement `cosine_similarity(a, b)` utility.
- [ ] 223. Implement `dot_product(a, b)` utility.
- [ ] 224. Ensure utilities auto-dispatch to WASM/WebGPU for large tensors.
- [ ] 225. Expose tensor shape manipulation (`view`, `reshape`, `transpose`).
- [ ] 226. Provide bi-directional conversion: `onnx9000.Tensor` <-> `Float32Array`.
- [ ] 227. Provide bi-directional conversion: `onnx9000.Tensor` <-> standard JSON arrays.
- [ ] 228. Handle multi-dimensional array slicing syntaxes in TS.
- [ ] 229. Support strided array access logic in JS wrappers.
- [ ] 230. Implement `Math.erf` polyfills if necessary.

### Phase 11: Export Tooling & Python Parity
- [ ] 231. Ensure Python API `onnx9000.transformers.pipeline()` matches JS API perfectly.
- [ ] 232. Implement auto-conversion script (`onnx9000 transformers export <model_id>`).
- [ ] 233. Generate `.onnx` files targeting optimal WebGPU topologies during export.
- [ ] 234. Generate optimized `tokenizer.json` files.
- [ ] 235. Extract and format `preprocessor_config.json`.
- [ ] 236. Extract and format `generation_config.json`.
- [ ] 237. Bundle pipeline configurations into `onnx9000-pipeline.json` for rapid loading.
- [ ] 238. Provide INT8 dynamic quantization during export.
- [ ] 239. Provide FP16 casting during export.
- [ ] 240. Publish an equivalent to `optimum-cli` natively within `onnx9000`.

### Phase 12: Worker & Web-Native Optimizations
- [ ] 241. Implement `WorkerPipeline` wrapper to execute pipelines entirely in a Web Worker.
- [ ] 242. Support Zero-Copy transfer of `Float32Array` buffers between main thread and workers.
- [ ] 243. Create message passing interface for streaming worker text generation.
- [ ] 244. Implement `SharedArrayBuffer` support for multi-threading if CORS/COOP allows.
- [ ] 245. Expose memory limit configurations (e.g., throwing error instead of crashing browser).
- [ ] 246. Support off-thread image decoding using `createImageBitmap`.
- [ ] 247. Prevent main thread blocking during large model compilation (WebGPU async pipeline creation).
- [ ] 248. Support Service Workers to preload pipelines for completely offline PWA experiences.
- [ ] 249. Integrate `requestIdleCallback` for non-blocking background model initialization.
- [ ] 250. Provide detailed performance tracing API (Network vs Compilation vs Inference time).

### Phase 13: Edge Case Handling
- [ ] 251. Handle inputs exceeding maximum sequence length gracefully (auto-truncation).
- [ ] 252. Manage WebGPU context loss and restore without application crash.
- [ ] 253. Handle completely empty text inputs.
- [ ] 254. Handle empty/zero-dimension images.
- [ ] 255. Catch and log unhandled exceptions securely without leaking internal paths.
- [ ] 256. Handle missing properties in older `config.json` revisions.
- [ ] 257. Provide graceful fallbacks for models without `generation_config`.
- [ ] 258. Support environments without IndexedDB (e.g., Incognito Mode).
- [ ] 259. Support environments without `fetch` API (Node.js fallback).
- [ ] 260. Manage circular dependencies in pipeline module loading.

### Phase 14: Quality Assurance & Testing
- [ ] 261. Achieve 100% API compatibility with Hugging Face's `transformers.js` v2/v3 syntax.
- [ ] 262. Create CI tests comparing Python HF outputs with TS `onnx9000` outputs.
- [ ] 263. Establish a daily test suite running against the top 100 HF models.
- [ ] 264. Unit test every tokenizer configuration option independently.
- [ ] 265. Unit test WASM image resizing against standard Pillow/OpenCV outputs.
- [ ] 266. Unit test WASM STFT/Mel outputs against librosa outputs.
- [ ] 267. Track memory leaks using Chrome DevTools automated puppeteer tests.
- [ ] 268. Maintain benchmarking dashboards comparing `onnx9000` vs `onnxruntime-web`.
- [ ] 269. Enforce strict TypeScript typing for all public APIs and Config objects.
- [ ] 270. Create interactive notebook tutorials (Jupyter/Observable) demonstrating usage.

### Phase 15: Developer Experience & Ecosystem
- [ ] 271. Provide React/Next.js boilerplate template using `onnx9000.transformers`.
- [ ] 272. Provide Vue/Nuxt boilerplate template.
- [ ] 273. Provide Chrome Extension boilerplate utilizing background scripts.
- [ ] 274. Create comprehensive documentation for migrating from `transformers.js`.
- [ ] 275. Support importing from CDNs (unpkg, jsdelivr) as an ES Module.
- [ ] 276. Build an interactive web playground (like HF Spaces) exclusively running `onnx9000`.
- [ ] 277. Implement a CLI tool to start a local REST API mimicking Hugging Face Inference Endpoints.
- [ ] 278. Implement a Node.js C++ addon bridge as a fallback for ultra-heavy models.
- [ ] 279. Support direct integration with `LangChain.js` tools and embeddings.
- [ ] 280. Integrate with the `Gradio` Python library.

### Phase 16: Extended Pipeline Features
- [ ] 281. Support returning probabilities for all classes in classification pipelines.
- [ ] 282. Add sentiment scores mapping (1 star to 5 star conversions).
- [ ] 283. Support multi-label classification post-processing (sigmoid instead of softmax).
- [ ] 284. Allow providing a custom `id2label` dictionary at runtime.
- [ ] 285. Implement context aggregation in Question Answering for long documents.
- [ ] 286. Handle batch generation padding dynamically.
- [ ] 287. Implement image tiling for high-resolution object detection.
- [ ] 288. Add vocal isolation/stemming capabilities to audio pipelines.
- [ ] 289. Add face detection specific utilities (wrapping generic object detection).
- [ ] 290. Support semantic search utilities (cosine similarity wrappers over feature extraction).

### Phase 17: Security & Reliability
- [ ] 291. Validate all model tensors to ensure bounds checking.
- [ ] 292. Implement a safe-loading mode that refuses to execute models with custom code.
- [ ] 293. Sandbox Web Workers executing untrusted user models.
- [ ] 294. Secure cache storage against cross-site scripting (XSS) extraction.
- [ ] 295. Implement resource-limit quotas for auto-downloading models.
- [ ] 296. Enforce strict Content Security Policy (CSP) guidelines in generated boilerplate.
- [ ] 297. Support offline-only mode (throwing errors instead of reaching out to network).
- [ ] 298. Validate digital signatures of official ONNX model binaries.
- [ ] 299. Prevent prototype pollution in configuration parsers.
- [ ] 300. Release v1.0 feature complete certification.
