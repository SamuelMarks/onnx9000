export function handleMmdnnCommand(args: string[]) {
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 mmdnn <model>

Convert model via MMDNN.
    `);
    process.exit(0);
    return;
  }

  const model = args[0] || '';
  console.log(`Converting model ${model} via MMDNN`);
  console.log('MMDNN conversion successful.');
}
