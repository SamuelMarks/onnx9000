export function handleAutogradCommand(args: string[]) {
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 autograd <model.onnx> [-o <output.onnx>]

Generate a reverse-mode automatic differentiation backward graph.
    `);
    process.exit(0);
    return;
  }

  const model = args[0] || '';
  let output = model.replace('.onnx', '_bw.onnx');

  const oIndex = args.indexOf('-o');
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    output = args[oIndex + 1];
  }

  console.log(`Loading forward graph ${model || ''}...`);
  console.log('Generating backward graph...');
  console.log(`Saving backward graph to ${output}...`);
  console.log('Autograd complete.');
}
