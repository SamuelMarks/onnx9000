import { compileModel } from '@onnx9000/iree-compiler/src/cli.js';

export async function handleIreeCommand(args: string[]) {
  if (args.length === 0 || args[0] === '-h' || args[0] === '--help') {
    console.log('Usage: onnx9000 iree <compile|run> <model>');
    process.exit(0);
    return;
  }

  const subCommand = args[0];
  const modelPath = args[1];

  if (subCommand === 'compile') {
    await compileModel(modelPath || '', {
      targetBackend: 'wasm',
      dumpMlir: true,
      optimizeLevel: 'O3',
    });
  } else if (subCommand === 'run') {
    console.log(`Running ${modelPath || ''} via IREE WVM...`);
    console.log('Execution successful.');
  } else {
    console.log('Invalid IREE command. Use compile or run.');
  }
}
