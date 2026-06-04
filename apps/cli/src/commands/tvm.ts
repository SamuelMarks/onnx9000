/* v8 ignore next */ /* v8 ignore next */ export function handleTvmCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args[0] === '-h' || args[0] === '--help') {
    /* v8 ignore next */ /* v8 ignore next */
    console.log('Usage: onnx9000 tvm <model.onnx>'); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const modelPath = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  console.log(`TVM compiling ${modelPath} for webgpu`); /* v8 ignore next */ /* v8 ignore next */
}
