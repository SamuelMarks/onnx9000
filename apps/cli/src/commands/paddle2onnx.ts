export function handlePaddle2ONNX(args: string[]) {
  if (args.length === 0) {
    console.error('Usage: onnx9000 paddle2onnx <model>');
    process.exit(1);
  }
  console.log(`Paddle2ONNX processed ${String(args[0])}`);
}
