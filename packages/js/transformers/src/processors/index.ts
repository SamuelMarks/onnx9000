export class AutoProcessor {
  static async fromPretrained(modelId: string): Promise<AutoProcessor> {
    return new AutoProcessor();
  }
  process(image: any): any {
    return { pixel_values: [0.5, 0.5] }; // mock
  }
}
