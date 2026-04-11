/* eslint-disable */
import { handleConvertCommand } from './commands/convert.js';
import { handleInspectCommand } from './commands/inspect.js';
import { handleOnnx2GgufCommand, handleGguf2OnnxCommand } from './commands/gguf.js';

async function main() {
  const args = process.argv.slice(2);
  if (args[0] === 'convert') {
    await handleConvertCommand(args.slice(1));
  } else if (args[0] === 'inspect') {
    await handleInspectCommand(args.slice(1));
  } else if (args[0] === 'onnx2gguf') {
    await handleOnnx2GgufCommand(args.slice(1));
  } else if (args[0] === 'gguf2onnx') {
    await handleGguf2OnnxCommand(args.slice(1));
  } else if (args[0] === 'serve') {
    const serveModule = await import('@onnx9000/serve');
    serveModule.runCli(args.slice(1));
  } else if (args[0] === 'array') {
    const arrayModule = await import('@onnx9000/array');
    console.log('Loaded array module:', !!arrayModule);
  } else if (args[0] === 'iree') {
    const ireeModule = await import('@onnx9000/iree-runtime');
    console.log('Loaded iree-runtime module:', !!ireeModule);
  } else if (args[0] === 'tensorrt') {
    const tensorrtModule = await import('@onnx9000/tensorrt');
    console.log('Loaded tensorrt module:', !!tensorrtModule);
  } else if (args[0] === 'triton') {
    const tritonModule = await import('@onnx9000/triton-compiler');
    console.log('Loaded triton-compiler module:', !!tritonModule);
  } else if (args[0] === 'openvino') {
    const ovModule = await import('@onnx9000/openvino-exporter');
    console.log('Loaded openvino-exporter module:', !!ovModule);
  } else {
    console.error('Usage: onnx9000 <command> [options]');
    console.error(
      'Available commands: convert, inspect, onnx2gguf, gguf2onnx, serve, array, iree, tensorrt, triton, openvino',
    );
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch(console.error);
}
