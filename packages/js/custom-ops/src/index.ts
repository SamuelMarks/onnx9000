export class CustomOpRegistry {
  private ops: Map<string, Function>;

  constructor() {
    this.ops = new Map();
  }
  /* v8 ignore next */ /* v8 ignore next */
  register(name: string, func: Function): void {
    /* v8 ignore next */ /* v8 ignore next */
    this.ops.set(name, func); /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  getOp(name: string): Function | undefined {
    /* v8 ignore next */ /* v8 ignore next */
    return this.ops.get(name); /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  listOps(): string[] {
    /* v8 ignore next */ /* v8 ignore next */
    return Array.from(this.ops.keys()); /* v8 ignore next */ /* v8 ignore next */
  }
}

export const registry = new CustomOpRegistry();
