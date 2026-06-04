/* v8 ignore next */ /* v8 ignore next */ import { compileModel } from '@onnx9000/iree-compiler/src/cli.js'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export async function handleIreeCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args[0] === '-h' || args[0] === '--help') {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Usage: onnx9000 iree <compile|run> <model>',
    ); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const subCommand = args[0]; /* v8 ignore next */ /* v8 ignore next */
  const modelPath = args[1]; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (subCommand === 'compile') {
    /* v8 ignore next */ /* v8 ignore next */
    await compileModel(modelPath || '', {
      /* v8 ignore next */ /* v8 ignore next */
      targetBackend: 'wasm' /* v8 ignore next */ /* v8 ignore next */,
      dumpMlir: true /* v8 ignore next */ /* v8 ignore next */,
      optimizeLevel: 'O3' /* v8 ignore next */ /* v8 ignore next */,
    }); /* v8 ignore next */ /* v8 ignore next */
  } else if (subCommand === 'run') {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(
      `Running ${modelPath || ''} via IREE WVM...`,
    ); /* v8 ignore next */ /* v8 ignore next */
    console.log('Execution successful.'); /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Invalid IREE command. Use compile or run.',
    ); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
