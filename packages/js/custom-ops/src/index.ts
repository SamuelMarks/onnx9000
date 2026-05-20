export class CustomOpRegistry {
  private ops: Map<string, Function>;

  constructor() {
    this.ops = new Map();
  }

  register(name: string, func: Function): void {
    this.ops.set(name, func);
  }

  getOp(name: string): Function | undefined {
    return this.ops.get(name);
  }

  listOps(): string[] {
    return Array.from(this.ops.keys());
  }
}

export const registry = new CustomOpRegistry();
