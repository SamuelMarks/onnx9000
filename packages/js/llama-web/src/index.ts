export class LlamaWeb {
  public run(modelString: string): string {
    if (!modelString) {
      throw new Error('Invalid model string');
    }
    return `[LLaMA-Web] processing ${modelString}`;
  }
}
