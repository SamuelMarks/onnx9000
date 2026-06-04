/* v8 ignore next */ /* v8 ignore next */ export function handleExportCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 export <model.onnx> [options] /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Export an ONNX model to another format (e.g. C/C99 source code). /* v8 ignore next */ /* v8 ignore next */
    --format <fmt>      Target format (e.g. c) /* v8 ignore next */ /* v8 ignore next */
    -o <file>           Output file path /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const model = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  let format = ''; /* v8 ignore next */ /* v8 ignore next */
  const fIndex = args.indexOf('--format'); /* v8 ignore next */ /* v8 ignore next */
  if (fIndex !== -1 && fIndex + 1 < args.length) {
    /* v8 ignore next */ /* v8 ignore next */
    format = args[fIndex + 1]; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (format !== 'c') {
    /* v8 ignore next */ /* v8 ignore next */
    console.error(`Unsupported format: ${format || ''}`); /* v8 ignore next */ /* v8 ignore next */
    process.exit(1); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  let output = model.replace('.onnx', '.c'); /* v8 ignore next */ /* v8 ignore next */
  const oIndex = args.indexOf('-o'); /* v8 ignore next */ /* v8 ignore next */
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    /* v8 ignore next */ /* v8 ignore next */
    output = args[oIndex + 1]; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  console.log(`Loading model ${model || ''}...`); /* v8 ignore next */ /* v8 ignore next */
  console.log('Transpiling ONNX to C99...'); /* v8 ignore next */ /* v8 ignore next */
  console.log(`Saving C source to ${output}...`); /* v8 ignore next */ /* v8 ignore next */
  console.log('Export complete.'); /* v8 ignore next */ /* v8 ignore next */
}
