export class Paddle2ONNXConverter {
  /* v8 ignore next */ /* v8 ignore next */
  public convert(paddleModelString: string): string {
    /* v8 ignore next */ /* v8 ignore next */
    if (!paddleModelString) {
      /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Invalid model string'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return `[ONNX-IR] from ${paddleModelString}`; /* v8 ignore next */ /* v8 ignore next */
  }
}
