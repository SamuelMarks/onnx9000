/* eslint-disable */
import { handleConvertCommand } from './commands/convert.js';
import { handleInspectCommand } from './commands/inspect.js';
import { handleOnnx2GgufCommand, handleGguf2OnnxCommand } from './commands/gguf.js';
import { handleJsonExtractCommand } from './commands/json-extract.js';
import { handlePytorchCodegenCommand } from './commands/pytorch-codegen.js';
import { handleWhisperLlmCommand } from './commands/whisper-llm.js';
import { handleLlamaWebCommand } from './commands/llama-web.js';
import { handleTfjsShimCommand } from './commands/tfjs-shim.js';
import { handleIreeCommand } from './commands/iree.js';
import { handleTritonCommand } from './commands/triton.js';
import { handleCoreMLCommand } from './commands/coreml.js';
import { handleTvmCommand } from './commands/tvm.js';
import { handleTensorRTCommand } from './commands/tensorrt.js';
import { handleDiffusersCommand } from './commands/diffusers.js';
import { handleTransformersCommand } from './commands/transformers.js';
import { handleEditCommand } from './commands/edit.js';
import { handleAutogradCommand } from './commands/autograd.js';
import { handleZooCommand } from './commands/zoo.js';
import { handleHummingbirdCommand } from './commands/hummingbird.js';
import { handleSparseCommand } from './commands/sparse.js';
import { handleOptimumCommand } from './commands/optimum.js';
import { handleSphinxDemoUICommand } from './commands/sphinx-demo-ui.js';
import { handleExportCommand } from './commands/export.js';
import { handleOnnx2TfCommand } from './commands/onnx2tf.js';
import { handleOptimizeCommand } from './commands/optimize.js';
import { handleSimplifyCommand } from './commands/simplify.js';
import { handleAgentCommand } from './commands/agent.js';
import { handleAppleCommand } from './commands/apple.js';
import { handleOnnx2cCommand } from './commands/onnx2c.js';
import { handleCudaCommand } from './commands/cuda.js';
import { handleJaxCommand } from './commands/jax.js';
import { handleMmdnnCommand } from './commands/mmdnn.js';
import { handleRocmCommand } from './commands/rocm.js';
import { handleWasmCommand } from './commands/wasm.js';
import { handleWebgpuCommand } from './commands/webgpu.js';
import { handleWebnnPolyfillCommand } from './commands/webnn-polyfill.js';
import { handleOnnxCheckerCommand } from './commands/onnx-checker.js';
import { handleScriptCommand } from './commands/script.js';

async function main() {
  const args = process.argv.slice(2);
  if (args[0] === 'convert') {
    await handleConvertCommand(args.slice(1));
  } else if (args[0] === 'inspect') {
    await handleInspectCommand(args.slice(1));
  } else if (args[0] === 'json-extract') {
    await handleJsonExtractCommand(args.slice(1));
  } else if (args[0] === 'pytorch-codegen') {
    await handlePytorchCodegenCommand(args.slice(1));
  } else if (args[0] === 'whisper-llm') {
    await handleWhisperLlmCommand(args.slice(1));
  } else if (args[0] === 'llama-web') {
    await handleLlamaWebCommand(args.slice(1));
  } else if (args[0] === 'tfjs-shim') {
    handleTfjsShimCommand(args.slice(1));
  } else if (args[0] === 'onnx2gguf') {
    await handleOnnx2GgufCommand(args.slice(1));
  } else if (args[0] === 'gguf2onnx') {
    await handleGguf2OnnxCommand(args.slice(1));
  } else if (args[0] === 'tvm') {
    handleTvmCommand(args.slice(1));
  } else if (args[0] === 'tensorrt') {
    handleTensorRTCommand(args.slice(1));
  } else if (args[0] === 'diffusers') {
    handleDiffusersCommand(args.slice(1));
  } else if (args[0] === 'serve') {
    const serveModule = await import('@onnx9000/serve');
    serveModule.runCli(args.slice(1));
  } else if (args[0] === 'array') {
    const arrayModule = await import('@onnx9000/array');
    console.log('Loaded array module:', !!arrayModule);
  } else if (args[0] === 'iree') {
    await handleIreeCommand(args.slice(1));
  } else if (args[0] === 'triton') {
    handleTritonCommand(args.slice(1));
  } else if (args[0] === 'openvino') {
    const ovModule = await import('@onnx9000/openvino-exporter');
    console.log('Loaded openvino-exporter module:', !!ovModule);
  } else if (args[0] === 'transformers') {
    await handleTransformersCommand(args.slice(1));
  } else if (args[0] === 'coreml') {
    await handleCoreMLCommand(args.slice(1));
  } else if (args[0] === 'edit') {
    await handleEditCommand(args.slice(1));
  } else if (args[0] === 'autograd') {
    await handleAutogradCommand(args.slice(1));
  } else if (args[0] === 'zoo') {
    await handleZooCommand(args.slice(1));
  } else if (args[0] === 'hummingbird') {
    await handleHummingbirdCommand(args.slice(1));
  } else if (args[0] === 'sparse') {
    await handleSparseCommand(args.slice(1));
  } else if (args[0] === 'optimum') {
    await handleOptimumCommand(args.slice(1));
  } else if (args[0] === 'sphinx-demo-ui') {
    await handleSphinxDemoUICommand(args.slice(1));
  } else if (args[0] === 'export') {
    await handleExportCommand(args.slice(1));
  } else if (args[0] === 'onnx2tf') {
    await handleOnnx2TfCommand(args.slice(1));
  } else if (args[0] === 'optimize') {
    await handleOptimizeCommand(args.slice(1));
  } else if (args[0] === 'simplify') {
    await handleSimplifyCommand(args.slice(1));
  } else if (args[0] === 'agent') {
    handleAgentCommand(args.slice(1));
  } else if (args[0] === 'apple') {
    handleAppleCommand(args.slice(1));
  } else if (args[0] === 'onnx2c') {
    handleOnnx2cCommand(args.slice(1));
  } else if (args[0] === 'cuda') {
    handleCudaCommand(args.slice(1));
  } else if (args[0] === 'jax') {
    handleJaxCommand(args.slice(1));
  } else if (args[0] === 'mmdnn') {
    handleMmdnnCommand(args.slice(1));
  } else if (args[0] === 'rocm') {
    handleRocmCommand(args.slice(1));
  } else if (args[0] === 'wasm') {
    handleWasmCommand(args.slice(1));
  } else if (args[0] === 'webgpu') {
    handleWebgpuCommand(args.slice(1));
  } else if (args[0] === 'webnn-polyfill') {
    handleWebnnPolyfillCommand(args.slice(1));
  } else if (args[0] === 'onnx-checker') {
    handleOnnxCheckerCommand(args.slice(1));
  } else if (args[0] === 'script') {
    handleScriptCommand(args.slice(1));
  } else {
    console.error('Usage: onnx9000 <command> [options]');
    console.error(
      'Available commands: convert, inspect, json-extract, pytorch-codegen, whisper-llm, llama-web, tfjs-shim, onnx2gguf, gguf2onnx, serve, array, iree, tensorrt, triton, openvino, transformers, coreml, edit, autograd, zoo, hummingbird, sparse, optimum, sphinx-demo-ui, export, onnx2tf, optimize, simplify, agent, apple, onnx2c, cuda, jax, mmdnn, onnx-checker, script',
    );
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch(console.error);
}
export { handleScriptCommand } from './commands/script.js';
