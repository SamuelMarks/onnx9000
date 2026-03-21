import { Program, Function, Block, Operation, Var } from './ast.js';
import { MILType, TensorType, ScalarType, TupleType, MILDataType } from './types.js';

export class Builder {
  private program = new Program();
  private currentFunction: Function | null = null;
  private currentBlock: Block | null = null;
  private varCounter = 0;

  createProgram(): Program {
    return this.program;
  }

  createFunction(name: string, inputs: Var[], outputs: Var[]): Function {
    const fn = new Function(name, inputs, outputs);
    this.program.addFunction(fn);
    this.currentFunction = fn;
    return fn;
  }

  createBlock(name: string): Block {
    if (!this.currentFunction) throw new Error('No active function');
    const block = new Block(name);
    this.currentFunction.addBlock(block);
    this.currentBlock = block;
    return block;
  }

  setBlock(block: Block) {
    this.currentBlock = block;
  }

  // Type system helpers
  tensor(dtype: MILDataType, shape: (number | string)[]): TensorType {
    return new TensorType(dtype, shape);
  }

  scalar(dtype: MILDataType): ScalarType {
    return new ScalarType(dtype);
  }

  tuple(elements: MILType[]): TupleType {
    return new TupleType(elements);
  }

  // Variable management
  createVar(name: string | null, type: MILType): Var {
    const varName = name || `%${this.varCounter++}`;
    return new Var(varName, type);
  }

  // Operations
  addOp(
    opType: string,
    inputs: Record<string, Var | Var[]>,
    outputs: Var[],
    attributes: Record<string, any> = {},
  ): Operation {
    if (!this.currentBlock) throw new Error('No active block');
    const op = new Operation(opType, inputs, outputs, attributes);
    this.currentBlock.addOperation(op);
    return op;
  }

  // Common MIL operations
  add(x: Var, y: Var, name: string | null = null): Var {
    // Basic shape inference: assume same shape
    const outVar = this.createVar(name, x.type);
    this.addOp('add', { x, y }, [outVar]);
    return outVar;
  }

  sub(x: Var, y: Var, name: string | null = null): Var {
    const outVar = this.createVar(name, x.type);
    this.addOp('sub', { x, y }, [outVar]);
    return outVar;
  }

  mul(x: Var, y: Var, name: string | null = null): Var {
    const outVar = this.createVar(name, x.type);
    this.addOp('mul', { x, y }, [outVar]);
    return outVar;
  }

  relu(x: Var, name: string | null = null): Var {
    const outVar = this.createVar(name, x.type);
    this.addOp('relu', { x }, [outVar]);
    return outVar;
  }
}
