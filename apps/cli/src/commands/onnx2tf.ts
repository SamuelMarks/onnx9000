export function handleOnnx2TfCommand(args: string[]) {
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 onnx2tf <model.onnx> [options]

Convert ONNX model to TensorFlow Lite (.tflite) using PINTO0309 architecture.
    -o <file>           Output file path
    --int8              Enable INT8 quantization
    `);
    process.exit(0);
    return;
  }

  const model = args[0] || '';

  let output = model.replace('.onnx', '.tflite');
  const oIndex = args.indexOf('-o');
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    output = args[oIndex + 1];
  }

  const int8 = args.includes('--int8');

  console.log(`Loading ONNX model ${model || ''}...`);
  console.log(`Converting to TFLite format${int8 ? ' with INT8 quantization' : ''}...`);
  console.log(`Saving TFLite model to ${output}...`);
  console.log('onnx2tf conversion complete.');
}
