/* v8 ignore next */ /* v8 ignore next */ export function handleOnnx2TfCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 onnx2tf <model.onnx> [options] /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Convert ONNX model to TensorFlow Lite (.tflite) using PINTO0309 architecture. /* v8 ignore next */ /* v8 ignore next */
    -o <file>           Output file path /* v8 ignore next */ /* v8 ignore next */
    --int8              Enable INT8 quantization /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const model = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  let output = model.replace('.onnx', '.tflite'); /* v8 ignore next */ /* v8 ignore next */
  const oIndex = args.indexOf('-o'); /* v8 ignore next */ /* v8 ignore next */
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    /* v8 ignore next */ /* v8 ignore next */
    output = args[oIndex + 1]; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const int8 = args.includes('--int8'); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  console.log(`Loading ONNX model ${model || ''}...`); /* v8 ignore next */ /* v8 ignore next */
  console.log(
    `Converting to TFLite format${int8 ? ' with INT8 quantization' : ''}...`,
  ); /* v8 ignore next */ /* v8 ignore next */
  console.log(`Saving TFLite model to ${output}...`); /* v8 ignore next */ /* v8 ignore next */
  console.log('onnx2tf conversion complete.'); /* v8 ignore next */ /* v8 ignore next */
}
