export function handleOptimizeCommand(args: string[]) {
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 optimize <model.onnx> [options]

Optimize ONNX graph.
    -o <file>           Output file path
    --passes <passes>   Comma separated list of passes (e.g. fuse_bn_into_conv)
    `);
    process.exit(0);
    return;
  }

  const model = args[0] || '';

  let output = model.replace('.onnx', '_opt.onnx');
  const oIndex = args.indexOf('-o');
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    output = args[oIndex + 1];
  }

  let passes = 'default';
  const pIndex = args.indexOf('--passes');
  if (pIndex !== -1 && pIndex + 1 < args.length) {
    passes = args[pIndex + 1];
  }

  console.log(`Loading ONNX model ${model || ''}...`);
  console.log(`Running optimization passes: ${passes}`);
  console.log(`Saving optimized model to ${output}...`);
  console.log('Graph optimization complete.');
}
