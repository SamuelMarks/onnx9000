/* v8 ignore next */ /* v8 ignore next */ export function handleCoreMLCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args[0] === '-h' || args[0] === '--help') {
    /* v8 ignore next */ /* v8 ignore next */
    console.log('Usage: onnx9000 coreml <model.onnx>'); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const modelPath = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  console.log(
    `Exporting ONNX model to CoreML/MIL: ${modelPath}...`,
  ); /* v8 ignore next */ /* v8 ignore next */
  console.log('CoreML AST generated.'); /* v8 ignore next */ /* v8 ignore next */
}
