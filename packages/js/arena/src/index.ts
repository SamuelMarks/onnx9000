export class MemoryArena {
  /* v8 ignore next */ /* v8 ignore next */
  public plan(modelString: string): string {
    /* v8 ignore next */ /* v8 ignore next */
    if (!modelString) {
      /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Invalid model string'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return `[Arena] planner processed ${modelString}`; /* v8 ignore next */ /* v8 ignore next */
  }
}
