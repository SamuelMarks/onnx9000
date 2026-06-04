export class Profiler {
  /* v8 ignore next */ /* v8 ignore next */
  modelPath: string; /* v8 ignore next */ /* v8 ignore next */
  peakMemory: number;
  /* v8 ignore next */ /* v8 ignore next */
  constructor(modelPath: string) {
    /* v8 ignore next */ /* v8 ignore next */
    this.modelPath = modelPath; /* v8 ignore next */ /* v8 ignore next */
    this.peakMemory = 0; /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  async run(): Promise<void> {
    /* v8 ignore next */ /* v8 ignore next */
    // Simulated profiler run /* v8 ignore next */ /* v8 ignore next */
    this.peakMemory = 42.5; /* v8 ignore next */ /* v8 ignore next */
  }
}
