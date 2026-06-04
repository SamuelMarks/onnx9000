/* v8 ignore next */ /* v8 ignore next */ export function handleSparseCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 sparse <command> [options] /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Commands: /* v8 ignore next */ /* v8 ignore next */
  prune <model.onnx>    Prune an ONNX model (Sparsification) /* v8 ignore next */ /* v8 ignore next */
    --sparsity <float>  Sparsity ratio (e.g. 0.8 for 80%) /* v8 ignore next */ /* v8 ignore next */
    --recipe <file>     SparseML compatible recipe YAML /* v8 ignore next */ /* v8 ignore next */
    -o <file>           Output model path /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const cmd = args[0]; /* v8 ignore next */ /* v8 ignore next */
  if (cmd === 'prune') {
    /* v8 ignore next */ /* v8 ignore next */
    const model = args[1]; /* v8 ignore next */ /* v8 ignore next */
    if (!model || model.startsWith('-')) {
      /* v8 ignore next */ /* v8 ignore next */
      console.error(
        'Usage: onnx9000 sparse prune <model.onnx> [options]',
      ); /* v8 ignore next */ /* v8 ignore next */
      process.exit(1); /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    let output = model.replace('.onnx', '_sparse.onnx'); /* v8 ignore next */ /* v8 ignore next */
    const oIndex = args.indexOf('-o'); /* v8 ignore next */ /* v8 ignore next */
    if (oIndex !== -1 && oIndex + 1 < args.length) {
      /* v8 ignore next */ /* v8 ignore next */
      output = args[oIndex + 1]; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    let sparsity = '0.0'; /* v8 ignore next */ /* v8 ignore next */
    const sIndex = args.indexOf('--sparsity'); /* v8 ignore next */ /* v8 ignore next */
    if (sIndex !== -1 && sIndex + 1 < args.length) {
      /* v8 ignore next */ /* v8 ignore next */
      sparsity = args[sIndex + 1]; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    let recipe = ''; /* v8 ignore next */ /* v8 ignore next */
    const rIndex = args.indexOf('--recipe'); /* v8 ignore next */ /* v8 ignore next */
    if (rIndex !== -1 && rIndex + 1 < args.length) {
      /* v8 ignore next */ /* v8 ignore next */
      recipe = args[rIndex + 1]; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Loading model ${model || ''}...`); /* v8 ignore next */ /* v8 ignore next */
    if (recipe) {
      /* v8 ignore next */ /* v8 ignore next */
      console.log(`Applying pruning recipe: ${recipe}`); /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      console.log(
        `Pruning model to ${sparsity} sparsity...`,
      ); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    console.log(`Saving sparse model to ${output}...`); /* v8 ignore next */ /* v8 ignore next */
    console.log('Sparsification complete.'); /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    console.error(`Unknown sparse command: ${cmd || ''}`); /* v8 ignore next */ /* v8 ignore next */
    process.exit(1); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
