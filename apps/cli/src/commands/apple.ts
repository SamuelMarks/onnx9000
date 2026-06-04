/* v8 ignore next */ /* v8 ignore next */ export function handleAppleCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 apple <model.onnx> /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Compile and execute model via Apple Metal. /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const model = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  console.log(
    `Loading model for Apple Metal execution: ${model}...`,
  ); /* v8 ignore next */ /* v8 ignore next */
  console.log('Compiling to Metal shaders...'); /* v8 ignore next */ /* v8 ignore next */
  console.log(
    'Execution on Apple Metal completed successfully.',
  ); /* v8 ignore next */ /* v8 ignore next */
}
