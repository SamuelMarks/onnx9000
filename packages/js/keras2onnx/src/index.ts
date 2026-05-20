export class Keras2ONNXConverter {
  public convert(kerasModelString: string): string {
    if (!kerasModelString) {
      throw new Error('Invalid model string');
    }
    return `[ONNX-IR] from keras ${kerasModelString}`;
  }
}
