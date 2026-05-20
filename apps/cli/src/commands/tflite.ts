export async function handleTfliteCommand(args: string[]) {
  // Alias to onnx2tf logic since they share the same underlying TFLite exporter.
  const { handleOnnx2TfCommand } = await import('./onnx2tf.js');
  handleOnnx2TfCommand(args);
}
