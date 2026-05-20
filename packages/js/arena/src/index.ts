export class MemoryArena {
  public plan(modelString: string): string {
    if (!modelString) {
      throw new Error('Invalid model string');
    }
    return `[Arena] planner processed ${modelString}`;
  }
}
