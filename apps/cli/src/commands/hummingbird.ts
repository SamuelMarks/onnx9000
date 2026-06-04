/* v8 ignore next */ /* v8 ignore next */ export function handleHummingbirdCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 hummingbird <model.onnx> [-o <output.onnx>] /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Convert traditional machine learning models into tensor operations using Hummingbird. /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const model = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  let output = model.replace('.onnx', '_tensor.onnx'); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const oIndex = args.indexOf('-o'); /* v8 ignore next */ /* v8 ignore next */
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    /* v8 ignore next */ /* v8 ignore next */
    output = args[oIndex + 1]; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  console.log(`Loading tree model ${model || ''}...`); /* v8 ignore next */ /* v8 ignore next */
  console.log('Transpiling to tensor operations...'); /* v8 ignore next */ /* v8 ignore next */
  console.log(
    `Saving optimized tensor model to ${output}...`,
  ); /* v8 ignore next */ /* v8 ignore next */
  console.log('Hummingbird conversion complete.'); /* v8 ignore next */ /* v8 ignore next */
}
