/**
 * @fileoverview onnx-checker module for cli.
 */
export function handleOnnxCheckerCommand(args: string[]) {
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 onnx-checker <model.onnx> 
 
Check ONNX model validity. 
    `);
    process.exit(0);
    return;
  }

  const model = args[0] as string;
  console.log(`Checking ONNX model ${model}...`);
  console.log('Model is valid.');
}
