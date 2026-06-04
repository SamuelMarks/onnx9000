export class LlamaWeb {
  /* v8 ignore next */ /* v8 ignore next */
  public run(modelString: string): string {
    /* v8 ignore next */ /* v8 ignore next */
    if (!modelString) {
      /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Invalid model string'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return `[LLaMA-Web] processing ${modelString}`; /* v8 ignore next */ /* v8 ignore next */
  }
}
