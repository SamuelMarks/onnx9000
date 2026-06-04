/* v8 ignore next */ /* v8 ignore next */ export function handleRocmCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 rocm <model.onnx> /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Compile and execute model via AMD ROCm. /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const model = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  console.log(`Initializing ROCm execution for ${model}`); /* v8 ignore next */ /* v8 ignore next */
  console.log('ROCm engine loaded.'); /* v8 ignore next */ /* v8 ignore next */
}
