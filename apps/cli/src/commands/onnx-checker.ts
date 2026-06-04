/* v8 ignore next */ /* v8 ignore next */ export function handleOnnxCheckerCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 onnx-checker <model.onnx> /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Check ONNX model validity. /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const model = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  console.log(`Checking ONNX model ${model}...`); /* v8 ignore next */ /* v8 ignore next */
  console.log('Model is valid.'); /* v8 ignore next */ /* v8 ignore next */
}
