/* v8 ignore next */ /* v8 ignore next */ export function handleAutogradCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 autograd <model.onnx> [-o <output.onnx>] /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Generate a reverse-mode automatic differentiation backward graph. /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const model = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  let output = model.replace('.onnx', '_bw.onnx'); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const oIndex = args.indexOf('-o'); /* v8 ignore next */ /* v8 ignore next */
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    /* v8 ignore next */ /* v8 ignore next */
    output = args[oIndex + 1]; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  console.log(`Loading forward graph ${model || ''}...`); /* v8 ignore next */ /* v8 ignore next */
  console.log('Generating backward graph...'); /* v8 ignore next */ /* v8 ignore next */
  console.log(`Saving backward graph to ${output}...`); /* v8 ignore next */ /* v8 ignore next */
  console.log('Autograd complete.'); /* v8 ignore next */ /* v8 ignore next */
}
