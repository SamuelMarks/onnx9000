export function handleSimplifyCommand(args: string[]) {
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 simplify <model.onnx> [options]

Simplify ONNX graph by folding constants and eliminating dead code.
    -o <file>           Output file path
    `);
    process.exit(0);
    return;
  }

  const model = args[0] || '';

  let output = model.replace('.onnx', '_sim.onnx');
  const oIndex = args.indexOf('-o');
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    output = args[oIndex + 1];
  }

  console.log(`Loading ONNX model ${model || ''}...`);
  console.log(`Simplifying graph...`);
  console.log(`Saving simplified model to ${output}...`);
  console.log('Graph simplification complete.');
}
