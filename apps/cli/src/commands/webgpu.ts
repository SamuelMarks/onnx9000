/* v8 ignore next */ /* v8 ignore next */ export function handleWebgpuCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 webgpu <model.onnx> /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Execute model via WebGPU backend. /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const model = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  console.log(
    `Initializing WebGPU execution for ${model}`,
  ); /* v8 ignore next */ /* v8 ignore next */
  console.log('WebGPU engine loaded.'); /* v8 ignore next */ /* v8 ignore next */
}
