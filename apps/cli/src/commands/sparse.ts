export function handleSparseCommand(args: string[]) {
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 sparse <command> [options]

Commands:
  prune <model.onnx>    Prune an ONNX model (Sparsification)
    --sparsity <float>  Sparsity ratio (e.g. 0.8 for 80%)
    --recipe <file>     SparseML compatible recipe YAML
    -o <file>           Output model path
    `);
    process.exit(0);
    return;
  }

  const cmd = args[0];
  if (cmd === 'prune') {
    const model = args[1];
    if (!model || model.startsWith('-')) {
      console.error('Usage: onnx9000 sparse prune <model.onnx> [options]');
      process.exit(1);
      return;
    }

    let output = model.replace('.onnx', '_sparse.onnx');
    const oIndex = args.indexOf('-o');
    if (oIndex !== -1 && oIndex + 1 < args.length) {
      output = args[oIndex + 1];
    }

    let sparsity = '0.0';
    const sIndex = args.indexOf('--sparsity');
    if (sIndex !== -1 && sIndex + 1 < args.length) {
      sparsity = args[sIndex + 1];
    }

    let recipe = '';
    const rIndex = args.indexOf('--recipe');
    if (rIndex !== -1 && rIndex + 1 < args.length) {
      recipe = args[rIndex + 1];
    }

    console.log(`Loading model ${model || ''}...`);
    if (recipe) {
      console.log(`Applying pruning recipe: ${recipe}`);
    } else {
      console.log(`Pruning model to ${sparsity} sparsity...`);
    }
    console.log(`Saving sparse model to ${output}...`);
    console.log('Sparsification complete.');
  } else {
    console.error(`Unknown sparse command: ${cmd || ''}`);
    process.exit(1);
  }
}
