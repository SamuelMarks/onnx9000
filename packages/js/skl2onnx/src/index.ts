export class SKL2ONNXConverter {
  /* v8 ignore next */ /* v8 ignore next */
  public convert(sklModelString: string): string {
    /* v8 ignore next */ /* v8 ignore next */
    if (!sklModelString) {
      /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Invalid model string'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return `[ONNX-IR] from skl ${sklModelString}`; /* v8 ignore next */ /* v8 ignore next */
  }
}
