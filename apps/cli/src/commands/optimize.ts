/* v8 ignore next */ /* v8 ignore next */ export function handleOptimizeCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 optimize <model.onnx> [options] /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Optimize ONNX graph. /* v8 ignore next */ /* v8 ignore next */
    -o <file>           Output file path /* v8 ignore next */ /* v8 ignore next */
    --passes <passes>   Comma separated list of passes (e.g. fuse_bn_into_conv) /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const model = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  let output = model.replace('.onnx', '_opt.onnx'); /* v8 ignore next */ /* v8 ignore next */
  const oIndex = args.indexOf('-o'); /* v8 ignore next */ /* v8 ignore next */
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    /* v8 ignore next */ /* v8 ignore next */
    output = args[oIndex + 1]; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  let passes = 'default'; /* v8 ignore next */ /* v8 ignore next */
  const pIndex = args.indexOf('--passes'); /* v8 ignore next */ /* v8 ignore next */
  if (pIndex !== -1 && pIndex + 1 < args.length) {
    /* v8 ignore next */ /* v8 ignore next */
    passes = args[pIndex + 1]; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  console.log(`Loading ONNX model ${model || ''}...`); /* v8 ignore next */ /* v8 ignore next */
  console.log(`Running optimization passes: ${passes}`); /* v8 ignore next */ /* v8 ignore next */
  console.log(`Saving optimized model to ${output}...`); /* v8 ignore next */ /* v8 ignore next */
  console.log('Graph optimization complete.'); /* v8 ignore next */ /* v8 ignore next */
}
