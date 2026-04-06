import { MILType, TensorType, ScalarType, TupleType } from './types.js';

export class Var {
  constructor(
    public name: string,
    public type: MILType,
  ) {}
}

export class Operation {
  constructor(
    public opType: string,
    public inputs: Record<string, Var | Var[]>,
    public outputs: Var[],
    public attributes: Record<string, ReturnType<typeof JSON.parse>> = {},
  ) {}
}

export class Block {
  public inputs: Var[] = [];
  public operations: Operation[] = [];
  public outputs: Var[] = [];

  constructor(public name: string) {}

  addOperation(op: Operation) {
    this.operations.push(op);
  }
}

export class Function {
  public blocks: Record<string, Block> = {};

  constructor(
    public name: string,
    public inputs: Var[],
    public outputs: Var[],
  ) {}

  addBlock(block: Block) {
    this.blocks[block.name] = block;
  }
}

export class Program {
  public functions: Record<string, Function> = {};

  addFunction(fn: Function) {
    this.functions[fn.name] = fn;
  }
}
