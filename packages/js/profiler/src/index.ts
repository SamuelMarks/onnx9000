export class Profiler {
  modelPath: string;
  peakMemory: number;

  constructor(modelPath: string) {
    this.modelPath = modelPath;
    this.peakMemory = 0;
  }

  async run(): Promise<void> {
    // Simulated profiler run
    this.peakMemory = 42.5;
  }
}
