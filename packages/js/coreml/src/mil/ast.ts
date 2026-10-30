/**
 * @fileoverview ast.ts
 * Provides ast functionality for the coreml package.
 */
import type { MILType } from './types.js';

/** Represents the `Var` class. */
export class Var {
  /**
   * Initializes a new `Var` instance.
   * @param name Parameter `name`.
   * @param type Parameter `type`.
   */
  constructor(
    public name: string,
    public type: MILType,
  ) {}
}

/** Represents the `Operation` class. */
export class Operation {
  /**
   * Initializes a new `Operation` instance.
   * @param opType Parameter `opType`.
   * @param inputs Parameter `inputs`.
   * @param outputs Parameter `outputs`.
   * @param attributes Parameter `attributes`.
   */
  constructor(
    public opType: string,
    public inputs: Record<string, Var | Var[]>,
    public outputs: Var[],
    public attributes: Record<string, ReturnType<typeof JSON.parse>> = {},
  ) {}
}

/** Represents the `Block` class. */
export class Block {
  /** Property `inputs` of `Block`. */
  public inputs: Var[] = [];
  /** Property `operations` of `Block`. */
  public operations: Operation[] = [];
  /** Property `outputs` of `Block`. */
  public outputs: Var[] = [];

  /**
   * Initializes a new `Block` instance.
   * @param name Parameter `name`.
   */
  constructor(public name: string) {}

  /**
   * Executes `addOperation` on `Block`.
   * @param op Parameter `op`.
   * @returns Result of `addOperation`.
   */
  addOperation(op: Operation) {
    this.operations.push(op);
  }
}

/** Represents the `MILFunction` class. */
export class MILFunction {
  /** Property `blocks` of `MILFunction`. */
  public blocks: Record<string, Block> = {};

  /**
   * Initializes a new `MILFunction` instance.
   * @param name Parameter `name`.
   * @param inputs Parameter `inputs`.
   * @param outputs Parameter `outputs`.
   */
  constructor(
    public name: string,
    public inputs: Var[],
    public outputs: Var[],
  ) {}

  /**
   * Executes `addBlock` on `MILFunction`.
   * @param block Parameter `block`.
   * @returns Result of `addBlock`.
   */
  addBlock(block: Block) {
    this.blocks[block.name] = block;
  }
}

/** Represents the `Program` class. */
export class Program {
  /** Property `functions` of `Program`. */
  public functions: Record<string, MILFunction> = {};

  /**
   * Executes `addFunction` on `Program`.
   * @param fn Parameter `fn`.
   * @returns Result of `addFunction`.
   */
  addFunction(fn: MILFunction) {
    this.functions[fn.name] = fn;
  }
}
