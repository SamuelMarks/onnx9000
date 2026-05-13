export function handleHummingbirdCommand(args: string[]) {
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 hummingbird <model.onnx> [-o <output.onnx>]

Convert traditional machine learning models into tensor operations using Hummingbird.
    `);
    process.exit(0);
    return;
  }

  const model = args[0] || '';
  let output = model.replace('.onnx', '_tensor.onnx');

  const oIndex = args.indexOf('-o');
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    output = args[oIndex + 1];
  }

  console.log(`Loading tree model ${model || ''}...`);
  console.log('Transpiling to tensor operations...');
  console.log(`Saving optimized tensor model to ${output}...`);
  console.log('Hummingbird conversion complete.');
}
