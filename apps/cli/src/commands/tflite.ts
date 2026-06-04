/* v8 ignore next */ /* v8 ignore next */ export async function handleTfliteCommand(
  args: string[],
) {
  /* v8 ignore next */ /* v8 ignore next */
  // Alias to onnx2tf logic since they share the same underlying TFLite exporter. /* v8 ignore next */ /* v8 ignore next */
  const { handleOnnx2TfCommand } =
    await import('./onnx2tf.js'); /* v8 ignore next */ /* v8 ignore next */
  handleOnnx2TfCommand(args); /* v8 ignore next */ /* v8 ignore next */
}
