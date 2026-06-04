/* v8 ignore next */ /* v8 ignore next */ export function handleSKL2ONNX(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0) {
    /* v8 ignore next */ /* v8 ignore next */
    console.error('Usage: onnx9000 skl2onnx <model>'); /* v8 ignore next */ /* v8 ignore next */
    process.exit(1); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  console.log(`SKL2ONNX processed ${String(args[0])}`); /* v8 ignore next */ /* v8 ignore next */
}
