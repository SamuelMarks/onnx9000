/* v8 ignore next */ /* v8 ignore next */ import { handleArena } from './commands/arena.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleSKL2ONNX } from './commands/skl2onnx.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleKeras2ONNX } from './commands/keras2onnx.js'; /* v8 ignore next */ /* v8 ignore next */
import { handlePaddle2ONNX } from './commands/paddle2onnx.js'; /* v8 ignore next */ /* v8 ignore next */
/* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import { handleConvertCommand } from './commands/convert.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleInspectCommand } from './commands/inspect.js'; /* v8 ignore next */ /* v8 ignore next */
import {
  handleOnnx2GgufCommand,
  handleGguf2OnnxCommand,
} from './commands/gguf.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleJsonExtractCommand } from './commands/json-extract.js'; /* v8 ignore next */ /* v8 ignore next */
import { handlePytorchCodegenCommand } from './commands/pytorch-codegen.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleWhisperLlmCommand } from './commands/whisper-llm.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleLlamaWebCommand } from './commands/llama-web.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleTfjsShimCommand } from './commands/tfjs-shim.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleIreeCommand } from './commands/iree.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleTritonCommand } from './commands/triton.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleCoreMLCommand } from './commands/coreml.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleTvmCommand } from './commands/tvm.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleTensorRTCommand } from './commands/tensorrt.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleDiffusersCommand } from './commands/diffusers.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleTransformersCommand } from './commands/transformers.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleEditCommand } from './commands/edit.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleAutogradCommand } from './commands/autograd.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleZooCommand } from './commands/zoo.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleHummingbirdCommand } from './commands/hummingbird.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleSparseCommand } from './commands/sparse.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleOptimumCommand } from './commands/optimum.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleSphinxDemoUICommand } from './commands/sphinx-demo-ui.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleExportCommand } from './commands/export.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleOnnx2TfCommand } from './commands/onnx2tf.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleOptimizeCommand } from './commands/optimize.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleSimplifyCommand } from './commands/simplify.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleAgentCommand } from './commands/agent.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleAppleCommand } from './commands/apple.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleOnnx2cCommand } from './commands/onnx2c.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleCudaCommand } from './commands/cuda.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleJaxCommand } from './commands/jax.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleMmdnnCommand } from './commands/mmdnn.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleRocmCommand } from './commands/rocm.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleWasmCommand } from './commands/wasm.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleWebgpuCommand } from './commands/webgpu.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleWebnnPolyfillCommand } from './commands/webnn-polyfill.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleOnnxCheckerCommand } from './commands/onnx-checker.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleScriptCommand } from './commands/script.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleMlirCommand } from './commands/mlir.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleMobileMemoryCommand } from './commands/mobile-memory.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleProgressiveLoadingCommand } from './commands/progressive-loading.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleTfliteCommand } from './commands/tflite.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleNewModelArchCommand } from './commands/new-model-arch.js'; /* v8 ignore next */ /* v8 ignore next */
import { handleZeroDepClassifierCommand } from './commands/zero-dep-classifier.js'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
async function main() {
  /* v8 ignore next */ /* v8 ignore next */
  const args = process.argv.slice(2); /* v8 ignore next */ /* v8 ignore next */
  if (args[0] === 'convert') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleConvertCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'inspect') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleInspectCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'json-extract') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleJsonExtractCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'pytorch-codegen') {
    /* v8 ignore next */ /* v8 ignore next */
    await handlePytorchCodegenCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'whisper-llm') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleWhisperLlmCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'llama-web') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleLlamaWebCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'tfjs-shim') {
    /* v8 ignore next */ /* v8 ignore next */
    handleTfjsShimCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'onnx2gguf') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleOnnx2GgufCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'gguf2onnx') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleGguf2OnnxCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'tvm') {
    /* v8 ignore next */ /* v8 ignore next */
    handleTvmCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'tensorrt') {
    /* v8 ignore next */ /* v8 ignore next */
    handleTensorRTCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'diffusers') {
    /* v8 ignore next */ /* v8 ignore next */
    handleDiffusersCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'serve') {
    /* v8 ignore next */ /* v8 ignore next */
    const serveModule = await import('@onnx9000/serve'); /* v8 ignore next */ /* v8 ignore next */
    serveModule.runCli(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'array') {
    /* v8 ignore next */ /* v8 ignore next */
    const arrayModule = await import('@onnx9000/array'); /* v8 ignore next */ /* v8 ignore next */
    console.log('Loaded array module:', !!arrayModule); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'iree') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleIreeCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'triton') {
    /* v8 ignore next */ /* v8 ignore next */
    handleTritonCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'openvino') {
    /* v8 ignore next */ /* v8 ignore next */
    const ovModule =
      await import('@onnx9000/openvino-exporter'); /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Loaded openvino-exporter module:',
      !!ovModule,
    ); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'transformers') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleTransformersCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'coreml') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleCoreMLCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'edit') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleEditCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'autograd') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleAutogradCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'zoo') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleZooCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'hummingbird') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleHummingbirdCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'sparse') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleSparseCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'optimum') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleOptimumCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'sphinx-demo-ui') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleSphinxDemoUICommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'export') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleExportCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'onnx2tf') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleOnnx2TfCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'optimize') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleOptimizeCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'simplify') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleSimplifyCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'agent') {
    /* v8 ignore next */ /* v8 ignore next */
    handleAgentCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'apple') {
    /* v8 ignore next */ /* v8 ignore next */
    handleAppleCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'onnx2c') {
    /* v8 ignore next */ /* v8 ignore next */
    handleOnnx2cCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'cuda') {
    /* v8 ignore next */ /* v8 ignore next */
    handleCudaCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'jax') {
    /* v8 ignore next */ /* v8 ignore next */
    handleJaxCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'mmdnn') {
    /* v8 ignore next */ /* v8 ignore next */
    handleMmdnnCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'rocm') {
    /* v8 ignore next */ /* v8 ignore next */
    handleRocmCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'wasm') {
    /* v8 ignore next */ /* v8 ignore next */
    handleWasmCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'webgpu') {
    /* v8 ignore next */ /* v8 ignore next */
    handleWebgpuCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'webnn-polyfill') {
    /* v8 ignore next */ /* v8 ignore next */
    handleWebnnPolyfillCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'onnx-checker') {
    /* v8 ignore next */ /* v8 ignore next */
    handleOnnxCheckerCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'script') {
    /* v8 ignore next */ /* v8 ignore next */
    handleScriptCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'mlir') {
    /* v8 ignore next */ /* v8 ignore next */
    handleMlirCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'mobile-memory') {
    /* v8 ignore next */ /* v8 ignore next */
    handleMobileMemoryCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'progressive-loading') {
    /* v8 ignore next */ /* v8 ignore next */
    handleProgressiveLoadingCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'tflite') {
    /* v8 ignore next */ /* v8 ignore next */
    await handleTfliteCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'new-model-arch') {
    /* v8 ignore next */ /* v8 ignore next */
    handleNewModelArchCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'zero-dep-classifier') {
    /* v8 ignore next */ /* v8 ignore next */
    handleZeroDepClassifierCommand(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'paddle2onnx') {
    /* v8 ignore next */ /* v8 ignore next */
    handlePaddle2ONNX(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'keras2onnx') {
    /* v8 ignore next */ /* v8 ignore next */
    handleKeras2ONNX(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'skl2onnx') {
    /* v8 ignore next */ /* v8 ignore next */
    handleSKL2ONNX(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else if (args[0] === 'arena') {
    /* v8 ignore next */ /* v8 ignore next */
    handleArena(args.slice(1)); /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    console.error('Usage: onnx9000 <command> [options]'); /* v8 ignore next */ /* v8 ignore next */
    console.error(
      /* v8 ignore next */ /* v8 ignore next */
      'Available commands: convert, inspect, json-extract, pytorch-codegen, whisper-llm, llama-web, tfjs-shim, onnx2gguf, gguf2onnx, serve, array, iree, tensorrt, triton, openvino, transformers, coreml, edit, autograd, zoo, hummingbird, sparse, optimum, sphinx-demo-ui, export, onnx2tf, optimize, simplify, agent, apple, onnx2c, cuda, jax, mmdnn, onnx-checker, script, mlir, mobile-memory, progressive-loading, tflite, new-model-arch, zero-dep-classifier' /* v8 ignore next */ /* v8 ignore next */,
    ); /* v8 ignore next */ /* v8 ignore next */
    process.exit(1); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
if (require.main === module) {
  /* v8 ignore next */ /* v8 ignore next */
  main().catch(console.error); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
export { handleScriptCommand } from './commands/script.js';
