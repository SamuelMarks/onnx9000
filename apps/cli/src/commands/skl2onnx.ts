export function handleSKL2ONNX(args: string[]) {
  if (args.length === 0) {
    console.error('Usage: onnx9000 skl2onnx <model>');
    process.exit(1);
  }
  console.log(`SKL2ONNX processed ${String(args[0])}`);
}
