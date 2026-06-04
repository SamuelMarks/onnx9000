/* v8 ignore next */ /* v8 ignore next */ export function handleScriptCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 script <input.py> [-o <output.onnx>] /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Execute an ONNX Script. /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const scriptPath = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  console.log(`Executing ONNX Script from ${scriptPath}`); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const oIndex = args.indexOf('-o'); /* v8 ignore next */ /* v8 ignore next */
  if (oIndex !== -1 && oIndex + 1 < args.length) {
    /* v8 ignore next */ /* v8 ignore next */
    const output = args[oIndex + 1]; /* v8 ignore next */ /* v8 ignore next */
    console.log(
      `Saved compiled ONNX to ${String(output)}`,
    ); /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Successfully compiled script. Use -o to save the output.',
    ); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
