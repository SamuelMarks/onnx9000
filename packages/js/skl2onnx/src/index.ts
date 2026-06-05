export class SKL2ONNXConverter {
  public convert(sklModelString: string): string {
    if (!sklModelString) {
      throw new Error('Invalid model string');
    }
    return `[ONNX-IR] from skl ${sklModelString}`;
  }
}
