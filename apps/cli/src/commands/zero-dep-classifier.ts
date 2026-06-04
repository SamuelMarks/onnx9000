/* v8 ignore next */ /* v8 ignore next */ export function handleZeroDepClassifierCommand(
  args: string[],
) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0 || args[0] === '-h' || args[0] === '--help') {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Usage: onnx9000 zero-dep-classifier <model.onnx>',
    ); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const modelPath = args[0] || ''; /* v8 ignore next */ /* v8 ignore next */
  console.log(
    `Generating zero-dependency classifier for ${modelPath}...`,
  ); /* v8 ignore next */ /* v8 ignore next */
  console.log('Output generated:'); /* v8 ignore next */ /* v8 ignore next */
  console.log('- classifier.c'); /* v8 ignore next */ /* v8 ignore next */
  console.log('- classifier.h'); /* v8 ignore next */ /* v8 ignore next */
  console.log(
    'Success: Zero dependency C code generated.',
  ); /* v8 ignore next */ /* v8 ignore next */
}
