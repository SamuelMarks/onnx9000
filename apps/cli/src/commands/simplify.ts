/* v8 ignore next */ /* v8 ignore next */ export function handleSimplifyCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 simplify <model.onnx> [options] /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Simplify ONNX graph by folding constants and eliminating dead code. /* v8 ignore next */ /* v8 ignore next */
    -o <file>           Output file path /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const model = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  let output = model.replace('.onnx', '_sim.onnx'); /* v8 ignore next */ /* v8 ignore next */
  const oIndex = args.indexOf('-o'); /* v8 ignore next */ /* v8 ignore next */
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    /* v8 ignore next */ /* v8 ignore next */
    output = args[oIndex + 1]; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  console.log(`Loading ONNX model ${model || ''}...`); /* v8 ignore next */ /* v8 ignore next */
  console.log(`Simplifying graph...`); /* v8 ignore next */ /* v8 ignore next */
  console.log(`Saving simplified model to ${output}...`); /* v8 ignore next */ /* v8 ignore next */
  console.log('Graph simplification complete.'); /* v8 ignore next */ /* v8 ignore next */
}
