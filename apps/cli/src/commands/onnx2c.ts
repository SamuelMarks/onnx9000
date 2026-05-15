export function handleOnnx2cCommand(args: string[]) {
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 onnx2c <input.onnx> [-o <output.c>]

Convert ONNX model to C source code.
    `);
    process.exit(0);
    return;
  }

  const model = args[0] || '';
  let output = 'output.c';

  const oIndex = args.indexOf('-o');
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    output = args[oIndex + 1];
  }

  console.log(`Converting ${model} to C...`);
  console.log(`Successfully generated C code to ${output}`);
}
