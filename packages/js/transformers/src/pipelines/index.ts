export class Pipeline {
  task: string;
  constructor(task: string) {
    this.task = task;
  }
  async run(input: string): Promise<any> {
    if (this.task === 'text-generation') {
      return [{ generated_text: input + ' [GENERATED]' }];
    }
    throw new Error(`Unsupported task: ${this.task}`);
  }
}
export function pipeline(task: string): Pipeline {
  return new Pipeline(task);
}
