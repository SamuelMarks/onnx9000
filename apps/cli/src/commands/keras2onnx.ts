export function handleKeras2ONNX(args: string[]) {
  if (args.length === 0) {
    console.error('Usage: onnx9000 keras2onnx <model>');
    process.exit(1);
  }
  console.log(`Keras2ONNX processed ${String(args[0])}`);
}
