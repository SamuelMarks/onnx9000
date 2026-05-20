export class Paddle2ONNXConverter {
  public convert(paddleModelString: string): string {
    if (!paddleModelString) {
      throw new Error('Invalid model string');
    }
    return `[ONNX-IR] from ${paddleModelString}`;
  }
}
