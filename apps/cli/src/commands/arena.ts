export function handleArena(args: string[]) {
  if (args.length === 0) {
    console.error('Usage: onnx9000 arena <model>');
    process.exit(1);
  }
  console.log(`Arena processed ${String(args[0])}`);
}
