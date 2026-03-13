# ONNX_NEXT: Gap Analysis for ml-switcheroo Integration & WASM Execution

This document analyzes the current gaps within the wider ONNX and ONNX-Runtime ecosystems that must be addressed to fulfill the vision of integrating `onnx9000` into `ml-switcheroo` as a standalone `pip install`able package, specifically focusing on enabling the planned Material 3 Stepper workflow for in-browser training and inference across multiple modalities.

**🔥 STATUS:** `onnx9000` is maturing rapidly. Over 200 standard ONNX operators have been implemented. The core IR parsing, autograd engine, static C++ memory planning, Apple Accelerate framework, WebAssembly SIMD backends, CUDA target, and control flow operators are fully integrated.


## The Vision
The target architecture involves:
1. Translating arbitrary ML frameworks to ONNX.
2. Providing a seamless UI (Stepper) to select modalities (Image, Video, Text, Audio/Mimetypes).
3. Allowing the user to choose between local/server execution or **in-browser training and serving via WASM/WebGPU**.
4. Providing real-time compilation/training logs and a live interactive system entirely within the browser.

## Ecosystem Gaps & Challenges

### 1. In-Browser Training Capabilities (WASM / WebGPU)
The most significant ecosystem gap is robust, out-of-the-box support for **training** ONNX models natively in the browser.
* **Inference-First Design:** ONNX and ONNX Runtime Web (ORT-Web) are overwhelmingly optimized for inference. While ONNX Runtime has a robust training API (ORT Training) for C++/Python, its WASM and WebGPU backends lack complete, officially supported APIs for backward passes, optimizers, and gradient accumulation in the browser environment.
* **Autograd Standardization:** Generating the backward graph (gradient ops) directly within the ONNX IR is historically fragmented. While `onnx9000` seems to address this locally with its `autograd` module, relying on standard ONNX ecosystem tools to dynamically build training graphs from arbitrary PyTorch/TF models for web execution is highly brittle.
* **Memory Constraints:** Training requires storing activations for the backward pass, significantly increasing memory usage. WASM currently has a rigid 4GB memory limit (until Wasm64 is fully adopted across all browsers), making in-browser training of modern multimodal models severely constrained.

### 2. Modality-Specific Data Pipelines in WASM
To support Image, Video, Text, and combined modalities in the browser without a persistent Python backend, the data preprocessing must happen efficiently in JS/WASM.
* **Missing Standardized WASM Tokenizers & Media Decoders:** The Python ecosystem relies on highly optimized C++ extensions (e.g., HuggingFace `tokenizers`, `torchvision`, `ffmpeg`) for preprocessing. While `onnxruntime-extensions` provides some operators (like basic text tokenization), comprehensive, fast, and lightweight in-browser data loaders for complex video and image streams are missing from the standard ONNX tooling.
* **Data I/O Operators:** ONNX lacks standardized operators for complex media decoding (e.g., "DecodeMP4" or "ExtractVideoFrames"), pushing this burden entirely to the frontend UI/JS layer before data can ever hit the ONNX graph.

### 3. Progressive Loading and Streaming
Loading large models for "live interactive systems" in the browser hits severe network and memory bottlenecks.
* **Monolithic File Format:** The `.onnx` file format generally requires downloading the entire protobuf graph and weights before initialization. There is no official ONNX ecosystem standard for streaming weights or progressive initialization, which is critical for serving large models in-browser without freezing the UI.
* **External Data Orchestration:** While ONNX supports external data (saving weights to separate `.bin` files), orchestrating the parallel fetch and progressive hydration of a WASM-based runtime is currently left entirely to custom frontend implementations.

### 4. Packaging and Python-to-WASM Build Pipelines
Integrating `onnx9000` as a `pip install`able package that seamlessly emits WASM components for the Sphinx interface presents structural challenges.
* **Cross-Compilation Complexity:** Creating a pip package that bundles pre-compiled WASM modules (or compiles them on the fly during user interaction) requires complex cross-platform build configurations. The ONNX ecosystem does not have a standardized "publish to PyPI with WASM targets" pipeline.
* **Pyodide/PyScript Limitations:** If the intention is to run the Python parts of `onnx9000` directly in the browser via Pyodide to generate ONNX graphs dynamically, the lack of full multithreading and the heavy initialization footprint of Pyodide add significant friction.

## Actionable Recommendations for `onnx9000`

To bypass these ecosystem gaps and successfully build the Material 3 Stepper workflow, `onnx9000` will need to implement specific architectural workarounds:

1. **AOT Backward Graph Generation:** Do not rely on ORT-Web for training algorithms. Instead, `onnx9000` should use its Python `autograd` and `codegen` capabilities to statically generate a monolithic ONNX graph that *explicitly contains* the forward pass, loss calculation, backward pass, and weight update steps as standard ONNX math operators. This single "training step" graph can then be executed by standard ORT-Web/WASM as a standard inference call.
2. **Bundle JS/WASM Preprocessing Bridges:** Ship lightweight JS/WASM data loaders for the target modalities (Text, Image, Video) alongside the `onnx9000` pip package, likely wrapping existing JS libraries (like WebCodecs API) rather than waiting for ONNX ecosystem support.
3. **WebGPU First:** Prioritize generating ONNX graphs compatible with the ORT WebGPU backend. CPU-based WASM will be too slow for anything beyond toy training examples in the browser.
4. **Chunked Export:** Ensure `onnx9000.export` can bundle the model, weights, and necessary preprocessing metadata into an optimized artifact specifically designed for chunked downloading and memory-mapped execution in the web frontend.


## Implementation Roadmap (300-Step Checklist)

### Phase 1: onnx9000 Core Packaging & CI/CD
- [x] [x] [x] **Step 1:** Create standard `pyproject.toml` with build-system and dependencies.
- [x] [x] [x] **Step 2:** Set up setuptools/flit/poetry build backend configuration.
- [x] [x] [x] **Step 3:** Refactor absolute paths to package-relative paths using `importlib.resources`.
- [x] [x] [x] **Step 4:** Set up GitHub Actions for automated Python testing and linting.
- [x] [x] [x] **Step 5:** Configure automated PyPI publishing workflows.
- [x] [x] [x] **Step 6:** Define public API exports in `src/onnx9000/__init__.py`.
- [x] [x] [x] **Step 7:** Add explicit type hinting to all public facing functions.
- [x] [x] [x] **Step 8:** Setup Sphinx documentation auto-generation.
- [x] [x] [x] **Step 9:** Integrate Sphinx build into CI/CD for GitHub Pages deployment.
- [x] [x] [x] **Step 10:** Create base CLI entry points in `pyproject.toml` (`onnx9000-cli`).
- [x] [x] [x] **Step 11:** Add comprehensive docstrings for the `autograd` module.
- [x] [x] [x] **Step 12:** Add comprehensive docstrings for the `codegen` module.
- [x] [x] [x] **Step 13:** Add comprehensive docstrings for the `frontend` module.
- [x] [x] [x] **Step 14:** Add comprehensive docstrings for the `export` module.
- [x] [x] [x] **Step 15:** Add comprehensive docstrings for the `runtime` module.
- [x] [x] [x] **Step 16:** Establish versioning scheme (e.g., semantic versioning).
- [x] [x] [x] **Step 17:** Write release notes template.
- [x] [x] [x] **Step 18:** Setup tox for multi-python environment testing (3.9 - 3.12).
- [x] [x] [x] **Step 19:** Ensure all test cases run cleanly from a pip-installed context.
- [x] [x] [x] **Step 20:** Audit and minimize third-party dependencies.
- [x] [x] [x] **Step 21:** Implement custom logging hierarchy replacing raw print statements.
- [x] [x] [x] **Step 22:** Create configuration overrides via environment variables.
- [x] [x] [x] **Step 23:** Create configuration overrides via global config objects.
- [x] [x] [x] **Step 24:** Implement a plugin architecture for custom operations.
- [x] [x] [x] **Step 25:** Publish alpha version 0.1.0 to TestPyPI.

### Phase 2: Autograd & AOT Training Graphs
- [x] [x] [x] **Step 26:** Audit existing VJP rules for mathematical correctness.
- [x] [x] [x] **Step 27:** Implement missing VJP rules for standard ONNX activation functions.
- [x] [x] [x] **Step 28:** Implement missing VJP rules for ONNX pooling operations.
- [x] [x] [x] **Step 29:** Implement missing VJP rules for ONNX convolution operations.
- [x] [x] [x] **Step 30:** Implement missing VJP rules for ONNX matrix multiplications.
- [x] [x] [x] **Step 31:** Implement missing VJP rules for ONNX normalization layers.
- [x] [x] [x] **Step 32:** Create an 'Optimizer' node generator (e.g., SGD).
- [x] [x] [x] **Step 33:** Create an 'Optimizer' node generator for Adam.
- [x] [x] [x] **Step 34:** Create an 'Optimizer' node generator for AdamW.
- [x] [x] [x] **Step 35:** Implement gradient accumulation logic within the ONNX IR.
- [x] [x] [x] **Step 36:** Inject loss calculation subgraphs (CrossEntropy, MSE).
- [x] [x] [x] **Step 37:** Implement dynamic parameter updates within the static graph.
- [x] [x] [x] **Step 38:** Handle shape inference for backward pass nodes.
- [x] [x] [x] **Step 39:** Write unit tests verifying AOT backward graph against PyTorch autograd.
- [x] [x] [x] **Step 40:** Optimize memory allocation in backward graph by re-using forward activations.
- [x] [x] [x] **Step 41:** Implement sub-graph extraction for partial model training.
- [x] [x] [x] **Step 42:** Support freezing specific layers by stripping their VJP rules.
- [x] [x] [x] **Step 43:** Implement gradient clipping nodes within the ONNX IR.
- [x] [x] [x] **Step 44:** Implement learning rate scheduling via ONNX inputs.
- [x] [x] [x] **Step 45:** Implement weight decay logic inside the AdamW subgraph.
- [x] [x] [x] **Step 46:** Ensure AOT graphs pass ONNX shape and type checker.
- [x] [x] [x] **Step 47:** Implement intermediate state saving for training checkpoints.
- [x] [x] [x] **Step 48:** Implement mixed precision (FP16/BF16) backward graph scaling.
- [x] [x] [x] **Step 49:** Support custom loss functions provided via PyTorch tracing.
- [x] [x] [x] **Step 50:** Validate AOT generated training step in ORT C++.

### Phase 3: Graph Optimization & IR Compliance
- [x] [x] [x] **Step 51:** Implement dead code elimination (DCE) for unneeded forward outputs.
- [x] [x] [x] **Step 52:** Implement constant folding for static training hyper-parameters.
- [x] [x] [x] **Step 53:** Implement operator fusion for adjacent linear + activation nodes.
- [x] [x] [x] **Step 54:** Ensure compliance with ONNX standard Opset 18+.
- [x] [x] [x] **Step 55:** Implement fallback mechanisms for unsupported older Opsets.
- [x] [x] [x] **Step 56:** Add validation passes to check for cycles in the DAG.
- [x] [x] [x] **Step 57:** Implement sub-graph flattening for nested ONNX structures.
- [x] [x] [x] **Step 58:** Optimize broadcast operations to minimize memory duplication.
- [x] [x] [x] **Step 59:** Implement memory layout transformations (NCHW <-> NHWC).
- [x] [x] [x] **Step 60:** Validate standard ONNX checker compliance on all output graphs.
- [x] [x] [x] **Step 61:** Implement debugging passes to inject Print/Identity nodes.
- [x] [x] [x] **Step 62:** Optimize graph for WebGPU constraints (e.g., max bind groups).
- [x] [x] [x] **Step 63:** Identify and replace WebGPU-unsupported ops with polyfills.
- [x] [x] [x] **Step 64:** Add graph visualization export (Netron compatible).
- [x] [x] [x] **Step 65:** Implement quantization aware training (QAT) nodes.
- [x] [x] [x] **Step 66:** Implement INT8 quantization for exported graphs.
- [x] [x] [x] **Step 67:** Implement dynamic shape support for variable batch sizes.
- [x] [x] [x] **Step 68:** Implement dynamic shape support for variable sequence lengths.
- [x] [x] [x] **Step 69:** Add specific handling for RNN/LSTM hidden states.
- [x] [x] [x] **Step 70:** Implement graph partitioning for multi-device execution.
- [x] [x] [x] **Step 71:** Create unit tests for graph optimizers.
- [x] [x] [x] **Step 72:** Benchmark optimized graphs against baseline PyTorch.
- [x] [x] [x] **Step 73:** Implement automated memory consumption estimation pass.
- [x] [x] [x] **Step 74:** Identify tensor life-cycles for WebGPU memory pooling.
- [x] [x] [x] **Step 75:** Finalize and lock the internal Graph IR data structures.

### Phase 4: Serialization & Progressive Loading
- [x] [x] [x] **Step 76:** Implement robust protobuf serialization for large (>2GB) graphs.
- [x] [x] [x] **Step 77:** Support exporting weights to external `.bin` files.
- [x] [x] [x] **Step 78:** Implement weight sharding (chunking) for HTTP progressive loading.
- [x] [x] [x] **Step 79:** Generate JSON manifests mapping chunks to tensor names.
- [x] [x] [x] **Step 80:** Implement int4/int8 weight compression for over-the-wire transfer.
- [x] [x] [x] **Step 81:** Add a metadata embedding pass (version, author, modality).
- [x] [x] [x] **Step 82:** Implement a progressive loading JS/WASM consumer.
- [x] [x] [x] **Step 83:** Design a file format wrapper to bundle ONNX + Preprocessing logic.
- [x] [x] [x] **Step 84:** Implement streaming graph parsing to avoid loading full protobuf into RAM.
- [x] [x] [x] **Step 85:** Support loading initial layer weights while background loading others.
- [x] [x] [x] **Step 86:** Implement integrity checks (SHA256) for chunked downloads.
- [x] [x] [x] **Step 87:** Add support for encrypting/decrypting weight chunks.
- [x] [x] [x] **Step 88:** Implement a caching layer in JS (IndexedDB) for downloaded chunks.
- [x] [x] [x] **Step 89:** Implement resume-capability for interrupted weight downloads.
- [x] [x] [x] **Step 90:** Add versioning to downloaded chunk caches to handle model updates.
- [x] [x] [x] **Step 91:** Create tools to merge split `.bin` files back into a monolithic file.
- [x] [x] [x] **Step 92:** Implement dynamic memory allocation for weights in WASM.
- [x] [x] [x] **Step 93:** Integrate with ONNX Runtime Web's external data loaders.
- [x] [x] [x] **Step 94:** Test progressive loading on slow network connections.
- [x] [x] [x] **Step 95:** Test progressive loading on memory-constrained devices (mobile).
- [x] [x] [x] **Step 96:** Implement a 'dry-run' load to verify structure without downloading weights.
- [x] [x] [x] **Step 97:** Create a visual progress-bar API for the frontend UI.
- [x] [x] [x] **Step 98:** Optimize protobuf parsing speed in the JS bridge.
- [x] [x] [x] **Step 99:** Ensure proper garbage collection of intermediate parsing buffers.
- [x] [x] [x] **Step 100:** Finalize the model export and chunking API.

### Phase 5: Web Worker Architecture
- [x] [x] [x] **Step 101:** Design the WebWorker topology (Main thread vs. Inference thread).
- [x] [x] [x] **Step 102:** Implement a message passing protocol (RPC) between Main and Worker.
- [x] [x] [x] **Step 103:** Set up SharedArrayBuffer for zero-copy tensor transfers.
- [x] [x] [x] **Step 104:** Implement cross-origin isolation headers for SharedArrayBuffer support.
- [x] [x] [x] **Step 105:** Wrap ONNX Runtime Web API in a robust WebWorker interface.
- [x] [x] [x] **Step 106:** Implement task queues for batched inference requests.
- [x] [x] [x] **Step 107:** Implement cancellation tokens for long-running training steps.
- [x] [x] [x] **Step 108:** Add heartbeat and health monitoring for the WASM worker.
- [x] [x] [x] **Step 109:** Implement automatic worker restarts on OOM or fatal WASM errors.
- [x] [x] [x] **Step 110:** Handle WASM instantiation asynchronously to avoid main thread blocking.
- [x] [x] [x] **Step 111:** Implement a web-worker pool for parallel preprocessing tasks.
- [x] [x] [x] **Step 112:** Support dynamic loading of different ORT Web backends (WASM, WebGL, WebGPU).
- [x] [x] [x] **Step 113:** Implement backend-fallback logic (WebGPU -> WASM).
- [x] [x] [x] **Step 114:** Add detailed telemetry and timing metrics to the worker RPC.
- [x] [x] [x] **Step 115:** Implement synchronous-like API wrappers for easy frontend usage.
- [x] [x] [x] **Step 116:** Write unit tests for the worker RPC protocol.
- [x] [x] [x] **Step 117:** Profile worker message passing latency.
- [x] [x] [x] **Step 118:** Optimize serialization of complex nested data structures over postMessage.
- [x] [x] [x] **Step 119:** Implement a 'training loop' directly inside the worker to minimize main-thread RPC.
- [x] [x] [x] **Step 120:** Add support for streaming outputs from the worker (e.g., text generation).
- [x] [x] [x] **Step 121:** Ensure memory leaks in the worker are prevented after tensor execution.
- [x] [x] [x] **Step 122:** Implement robust error serialization from worker to main thread.
- [x] [x] [x] **Step 123:** Support loading custom ONNX Runtime Web extensions.
- [x] [x] [x] **Step 124:** Add integration tests across multiple browser engines (V8, SpiderMonkey, JavaScriptCore).
- [x] [x] [x] **Step 125:** Document the WebWorker API architecture.

### Phase 6: WebGPU Computation & Memory
- [x] [x] [x] **Step 126:** Initialize WebGPU device and standard command queues.
- [x] [x] [x] **Step 127:** Verify browser WebGPU support and handle gracefully if missing.
- [x] [x] [x] **Step 128:** Configure standard WebGPU bind groups for ONNX tensors.
- [x] [x] [x] **Step 129:** Implement memory pooling for intermediate tensors during the backward pass.
- [x] [x] [x] **Step 130:** Handle the WebGPU 4GB memory limits proactively.
- [x] [x] [x] **Step 131:** Map ONNX tensor shapes to WebGPU buffer dimensions.
- [x] [x] [x] **Step 132:** Implement custom WGSL shaders for operations missing in standard ORT-Web.
- [x] [x] [x] **Step 133:** Implement optimized matrix multiplication (GEMM) in WGSL.
- [x] [x] [x] **Step 134:** Implement parallel reductions in WGSL.
- [x] [x] [x] **Step 135:** Implement fused activation WGSL shaders.
- [x] [x] [x] **Step 136:** Profile WebGPU memory bandwidth utilization.
- [x] [x] [x] **Step 137:** Implement buffer mapping/unmapping for reading results back to JS.
- [x] [x] [x] **Step 138:** Optimize the CPU-to-GPU data transfer bottleneck.
- [x] [x] [x] **Step 139:** Handle WebGPU device loss and context recreation.
- [x] [x] [x] **Step 140:** Support mixed precision (FP16) compute via `shader-f16` WebGPU extension.
- [x] [x] [x] **Step 141:** Implement WebGPU compute pipeline caching.
- [x] [x] [x] **Step 142:** Implement asynchronous shader compilation.
- [x] [x] [x] **Step 143:** Add debug markers and labels to WebGPU objects for profiling.
- [x] [x] [x] **Step 144:** Write unit tests for custom WGSL implementations.
- [x] [x] [x] **Step 145:** Benchmark WebGPU training against native PyTorch CPU.
- [x] [x] [x] **Step 146:** Implement dynamic dispatch sizing based on tensor shapes.
- [x] [x] [x] **Step 147:** Optimize memory alignment for WebGPU storage buffers.
- [x] [x] [x] **Step 148:** Implement staging buffers for efficient data uploads.
- [x] [x] [x] **Step 149:** Handle limitations on max storage buffer binding size.
- [x] [x] [x] **Step 150:** Finalize WebGPU backend integration.

### Phase 7: Modality Pipeline - Text
- [x] [x] [x] **Step 151:** Implement a generic Byte-Pair Encoding (BPE) tokenizer in JS.
- [x] [x] [x] **Step 152:** Implement WordPiece tokenization in JS.
- [x] [x] [x] **Step 153:** Implement SentencePiece tokenization in JS.
- [x] [x] [x] **Step 154:** Support loading HuggingFace `tokenizer.json` files directly.
- [x] [x] [x] **Step 155:** Optimize tokenizer performance using WebWorkers.
- [x] [x] [x] **Step 156:** Implement chunking and overlapping for long documents.
- [x] [x] [x] **Step 157:** Implement padding and truncation to fixed sequence lengths.
- [x] [x] [x] **Step 158:** Generate attention masks for standard transformer models.
- [x] [x] [x] **Step 159:** Generate token type IDs for BERT-like models.
- [x] [x] [x] **Step 160:** Implement special token (BOS, EOS, PAD, UNK) injection.
- [x] [x] [x] **Step 161:** Support dynamic vocabulary resizing.
- [x] [x] [x] **Step 162:** Implement batching logic for multiple text inputs.
- [x] [x] [x] **Step 163:** Add detokenization (ID to string) capabilities.
- [x] [x] [x] **Step 164:** Profile text tokenization latency in browser.
- [x] [x] [x] **Step 165:** Write extensive test cases comparing JS tokenizer outputs to Python.
- [x] [x] [x] **Step 166:** Implement text stream parsing (e.g., for SSE inputs).
- [x] [x] [x] **Step 167:** Add support for custom normalization rules (unicode, lowercasing).
- [x] [x] [x] **Step 168:** Add support for pre-tokenization regex splits.
- [x] [x] [x] **Step 169:** Implement efficient vocabulary lookup tables.
- [x] [x] [x] **Step 170:** Optimize memory usage for very large vocabularies (>100k tokens).
- [x] [x] [x] **Step 171:** Add an interface for custom text preprocessing plugins.
- [x] [x] [x] **Step 172:** Implement caching for frequently tokenized text.
- [x] [x] [x] **Step 173:** Build a tokenizer visualization tool for debugging.
- [x] [x] [x] **Step 174:** Integrate text pipeline with the WebWorker architecture.
- [x] [x] [x] **Step 175:** Finalize Text modality API.

### Phase 8: Modality Pipeline - Image
- [x] [x] [x] **Step 176:** Implement standardized Image loading using the `Image` and `canvas` APIs.
- [x] [x] [x] **Step 177:** Extract raw RGBA pixel data efficiently.
- [x] [x] [x] **Step 178:** Implement bilinear and bicubic image resizing in JS/WebGL.
- [x] [x] [x] **Step 179:** Implement center cropping and random cropping.
- [x] [x] [x] **Step 180:** Implement channel reordering (RGBA to RGB, HWC to CHW).
- [x] [x] [x] **Step 181:** Implement image normalization (mean/std subtraction).
- [x] [x] [x] **Step 182:** Support converting `Uint8ClampedArray` to `Float32Array` efficiently.
- [x] [x] [x] **Step 183:** Implement image batching for training/inference.
- [x] [x] [x] **Step 184:** Handle aspect ratio preservation during resizing.
- [x] [x] [x] **Step 185:** Support loading base64 data URIs.
- [x] [x] [x] **Step 186:** Support direct processing of `File` and `Blob` objects.
- [x] [x] [x] **Step 187:** Implement basic data augmentation in JS (flip, rotate).
- [x] [x] [x] **Step 188:** Implement color jittering in JS.
- [x] [x] [x] **Step 189:** Add support for HDR and 16-bit image formats if available.
- [x] [x] [x] **Step 190:** Use OffscreenCanvas for WebWorker-based image processing.
- [x] [x] [x] **Step 191:** Implement WebGL-accelerated image preprocessing.
- [x] [x] [x] **Step 192:** Profile image loading and preprocessing pipeline.
- [x] [x] [x] **Step 193:** Write unit tests comparing image tensors to `torchvision.transforms`.
- [x] [x] [x] **Step 194:** Implement generic handling for different color spaces (HSV, Grayscale).
- [x] [x] [x] **Step 195:** Add a caching layer for preprocessed image tensors.
- [x] [x] [x] **Step 196:** Implement progressive image loading support.
- [x] [x] [x] **Step 197:** Integrate image pipeline with the WebWorker architecture.
- [x] [x] [x] **Step 198:** Build a visual image debugger for inspecting tensor data.
- [x] [x] [x] **Step 199:** Finalize Image modality API.
- [x] [x] [x] **Step 200:** Ensure memory is freed after extracting image data.

### Phase 9: Modality Pipeline - Video & Audio
- [x] [x] [x] **Step 201:** Integrate with the WebCodecs API for hardware-accelerated video decoding.
- [x] [x] [x] **Step 202:** Extract specific frames from MP4/WebM video streams.
- [x] [x] [x] **Step 203:** Implement frame batching into a 4D tensor (T, C, H, W).
- [x] [x] [x] **Step 204:** Handle variable frame rates and temporal subsampling.
- [x] [x] [x] **Step 205:** Implement video stream buffering for continuous inference.
- [x] [x] [x] **Step 206:** Optimize WebCodecs frame data extraction to `Float32Array`.
- [x] [x] [x] **Step 207:** Integrate with the Web Audio API for decoding audio files (MP3, WAV).
- [x] [x] [x] **Step 208:** Implement waveform extraction to `Float32Array`.
- [x] [x] [x] **Step 209:** Implement Short-Time Fourier Transform (STFT) in JS/WASM.
- [x] [x] [x] **Step 210:** Implement Mel-Spectrogram generation matching `torchaudio`.
- [x] [x] [x] **Step 211:** Implement audio resampling to target sample rates.
- [x] [x] [x] **Step 212:** Implement audio normalization and padding.
- [x] [x] [x] **Step 213:** Handle streaming audio directly from the user's microphone (`getUserMedia`).
- [x] [x] [x] **Step 214:** Handle streaming video directly from the user's camera.
- [x] [x] [x] **Step 215:** Implement chunked processing for long audio/video files.
- [x] [x] [x] **Step 216:** Write unit tests verifying spectrogram outputs.
- [x] [x] [x] **Step 217:** Profile audio and video extraction pipelines.
- [x] [x] [x] **Step 218:** Implement synchronization between audio and video streams (multimodal).
- [x] [x] [x] **Step 219:** Add support for text-to-speech outputs via Web Audio API.
- [x] [x] [x] **Step 220:** Add support for video generation/rendering outputs.
- [x] [x] [x] **Step 221:** Implement memory limits to prevent out-of-memory on large videos.
- [x] [x] [x] **Step 222:** Integrate AV pipelines with the WebWorker architecture.
- [x] [x] [x] **Step 223:** Build visual debuggers for video frames and audio waveforms.
- [x] [x] [x] **Step 224:** Ensure proper resource disposal for WebCodecs and Web Audio context.
- [x] [x] [x] **Step 225:** Finalize Video and Audio modality APIs.

### Phase 10: Material 3 Stepper UI Foundation
- [x] [x] [x] **Step 226:** Initialize Vanilla JS frontend scaffold.
- [x] [x] [x] **Step 227:** Integrate Material Design 3 (M3) component library.
- [x] [x] [x] **Step 228:** Set up the overarching global state management (Redux, Zustand, or Context).
- [x] [x] [x] **Step 229:** Implement the main Stepper layout container.
- [x] [x] [x] **Step 230:** Design the global theming (Dark/Light mode) based on M3 specs.
- [x] [x] [x] **Step 231:** Implement Step 0 layout container.
- [x] [x] [x] **Step 232:** Implement Step 1 layout container.
- [x] [x] [x] **Step 233:** Implement Step 1a layout container.
- [x] [x] [x] **Step 234:** Implement Step 2 layout container.
- [x] [x] [x] **Step 235:** Implement Step 3 layout container.
- [x] [x] [x] **Step 236:** Set up routing or internal state transitions between steps.
- [x] [x] [x] **Step 237:** Add transition animations between stepper phases.
- [x] [x] [x] **Step 238:** Implement persistent state saving to `localStorage` (recover on refresh).
- [x] [x] [x] **Step 239:** Create standard dialog/modal components for configuration.
- [x] [x] [x] **Step 240:** Create standard alert/toast notification components.
- [x] [x] [x] **Step 241:** Build a centralized error boundary for catching frontend crashes.
- [x] [x] [x] **Step 242:** Implement responsive design for mobile and desktop views.
- [x] [x] [x] **Step 243:** Add accessibility (a11y) labels and ARIA attributes.
- [x] [x] [x] **Step 244:** Integrate a syntax-highlighting code editor component (e.g., Monaco).
- [x] [x] [x] **Step 245:** Create a visual directed acyclic graph (DAG) viewer for ONNX models.
- [x] [x] [x] **Step 246:** Implement Step 0 visual mapping ('Python ML framework et al.').
- [x] [x] [x] **Step 247:** Load demo python code into the Step 0 code editor.
- [x] [x] [x] **Step 248:** Mock the initial Python-to-ONNX translation in the UI.
- [x] [x] [x] **Step 249:** Implement the 'Next' and 'Back' stepper control logic.
- [x] [x] [x] **Step 250:** Finalize the base M3 application shell.

### Phase 11: UI Step 1 & 1a (Modality & Execution)
- [x] [x] [x] **Step 251:** Design the split-view layout: Last RH side of Step 0 vs. ONNX representation.
- [x] [x] [x] **Step 252:** Implement the ONNX DAG rendering on the right-hand side.
- [x] [x] [x] **Step 253:** Enable interactive node inspection in the ONNX graph view.
- [x] [x] [x] **Step 254:** Highlight translated layers mapping Python code to ONNX nodes.
- [x] [x] [x] **Step 255:** Implement the 'Modality' dropdown selector.
- [x] [x] [x] **Step 256:** Add UI for 'Image' modality configuration.
- [x] [x] [x] **Step 257:** Add UI for 'Video' modality configuration.
- [x] [x] [x] **Step 258:** Add UI for 'Text' modality configuration.
- [x] [x] [x] **Step 259:** Add UI for 'Image+text' (Multimodal) configuration.
- [x] [x] [x] **Step 260:** Add UI for 'Mimetypes' / Audio configuration.
- [x] [x] [x] **Step 261:** Implement the execution environment toggle (Local/Server vs. Browser).
- [x] [x] [x] **Step 262:** Add explanations/tooltips detailing the memory limits of Browser execution.
- [x] [x] [x] **Step 263:** Implement the dataset upload/selection interface.
- [x] [x] [x] **Step 264:** Support drag-and-drop for local dataset folders.
- [x] [x] [x] **Step 265:** Implement dataset preview components (image gallery, text samples).
- [x] [x] [x] **Step 266:** Add configuration forms for training hyperparameters (LR, batch size, epochs).
- [x] [x] [x] **Step 267:** Validate user configuration before allowing progression to Step 2.
- [x] [x] [x] **Step 268:** Mock the transition state while 'compiling' the model for the web.
- [x] [x] [x] **Step 269:** Integrate `onnx9000` Pyodide bindings to dynamically parse requested models.
- [x] [x] [x] **Step 270:** Finalize Step 1 & 1a UI and logic flows.

### Phase 12: UI Step 2 (Logs & Transport)
- [x] [x] [x] **Step 271:** Implement the real-time compilation/training log terminal window.
- [x] [x] [x] **Step 272:** Integrate terminal UI (e.g., Xterm.js) for realistic log rendering.
- [x] [x] [x] **Step 273:** Stream build outputs from Pyodide/WebWorker to the terminal.
- [x] [x] [x] **Step 274:** Implement a visual progress bar for weight downloading.
- [x] [x] [x] **Step 275:** Implement a visual progress bar for model compilation (WebGPU pipelines).
- [x] [x] [x] **Step 276:** Display current memory usage and WebGPU context status.
- [x] [x] [x] **Step 277:** Implement a toggle to download the compiled model to local disk.
- [x] [x] [x] **Step 278:** Add a feature to export the entire JS/HTML package for local hosting.
- [x] [x] [x] **Step 279:** Implement 'Serve in browser' start button logic.
- [x] [x] [x] **Step 280:** Implement the training loop visualizer (Loss vs. Epoch line charts).
- [x] [x] [x] **Step 281:** Stream training metrics from the WebWorker to the charts.
- [x] [x] [x] **Step 282:** Implement early-stopping controls in the UI.
- [x] [x] [x] **Step 283:** Add 'Pause' and 'Resume' controls for the browser training loop.
- [x] [x] [x] **Step 284:** Implement automatic error logging and diagnostics displays.
- [x] [x] [x] **Step 285:** Add a button to export the training logs as a text file.
- [x] [x] [x] **Step 286:** Finalize the logic for routing execution based on Step 1a choices.
- [x] [x] [x] **Step 287:** Test UI resilience during heavy main-thread blocking (ensure UI doesn't freeze).
- [x] [x] [x] **Step 288:** Optimize chart rendering performance for high-frequency updates.
- [x] [x] [x] **Step 289:** Finalize Step 2 UI and state management.

### Phase 13: UI Step 3 (Live Execution & Export)
- [x] [x] [x] **Step 290:** Design the 'Live System' interactive dashboard layout.
- [x] [x] [x] **Step 291:** Implement input interfaces specific to the chosen modality (e.g., text chat, image upload).
- [x] [x] [x] **Step 292:** Hook up UI inputs to the WebWorker inference pipeline.
- [x] [x] [x] **Step 293:** Implement real-time rendering of inference outputs.
- [x] [x] [x] **Step 294:** Add a latency/performance metrics overlay for live inference.
- [x] [x] [x] **Step 295:** Implement a feature to switch between the base model and the newly trained model.
- [x] [x] [x] **Step 296:** Implement a toggle to continuously train on new interactions.
- [x] [x] [x] **Step 297:** Add functionality to 'Save Weights' - extracting the updated tensors from WebGPU.
- [x] [x] [x] **Step 298:** Convert extracted weights back to ONNX `.bin` or standard checkpoint formats.
- [x] [x] [x] **Step 299:** Implement a download trigger for the trained `.onnx` model.
- [x] [x] [x] **Step 300:** Provide code snippets on how to load the downloaded model back into PyTorch.
- [x] [x] [x] **Step 301:** Implement a 'Reset to Base' functionality.
- [x] [x] [x] **Step 302:** Test the live system with text generation (LLM style output streaming).
- [x] [x] [x] **Step 303:** Test the live system with image classification outputs.
- [x] [x] [x] **Step 304:** Test the live system with a toy multimodal task.
- [x] [x] [x] **Step 305:** Finalize Step 3 UI and interactivity.
- [x] [x] [x] **Step 306:** Ensure all WebWorker and WebGPU resources are cleanly disposed of on exit.

### Phase 14: ml-switcheroo Integration & Polish
- [x] [x] [x] **Step 307:** Install `onnx9000` via pip into the `ml-switcheroo` environment.
- [x] [x] [x] **Step 308:** Replace hardcoded paths in `ml-switcheroo` with dynamic `onnx9000` imports.
- [x] [x] [x] **Step 309:** Integrate the generated Material 3 frontend into the `ml-switcheroo` Sphinx docs/WASM interface.
- [x] [x] [x] **Step 310:** Ensure proper routing from Sphinx static pages to the dynamic Stepper app.
- [x] [x] [x] **Step 311:** Test end-to-end integration: Python definition -> ONNX compilation -> Browser UI -> WebGPU Training.
- [x] [x] [x] **Step 312:** Fix cross-origin resource sharing (CORS) issues for local testing.
- [x] [x] [x] **Step 313:** Optimize bundle size for the frontend application.
- [x] [x] [x] **Step 314:** Minify and obfuscate production JS builds.
- [x] [x] [x] **Step 315:** Write integration tests validating the full `ml-switcheroo` + `onnx9000` flow.
- [x] [x] [x] **Step 316:** Create demo videos or GIFs for the README.
- [x] [x] [x] **Step 317:** Update `ARCHITECTURE.md` to reflect the new pipeline.
- [x] [x] [x] **Step 318:** Update `ROADMAP.md`.
- [x] [x] [x] **Step 319:** Finalize `ONNX_NEXT.md` documentation.
- [x] [x] [x] **Step 320:** Conduct a final code review for Python and JS codebases.
- [x] [x] [x] **Step 321:** Release final integrated version 1.0.0.

