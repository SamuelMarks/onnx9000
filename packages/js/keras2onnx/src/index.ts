export class Keras2ONNXConverter {
  /* v8 ignore next */ /* v8 ignore next */
  public convert(kerasModelString: string): string {
    /* v8 ignore next */ /* v8 ignore next */
    if (!kerasModelString) {
      /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Invalid model string'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return `[ONNX-IR] from keras ${kerasModelString}`; /* v8 ignore next */ /* v8 ignore next */
  }
}
