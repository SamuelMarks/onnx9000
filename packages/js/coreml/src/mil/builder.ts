/**
 * @fileoverview builder.ts
 * Provides builder functionality for the coreml package.
 */
import { Onnx9000Error } from '@onnx9000/core';
import { Block, MILFunction, Operation, Program, Var } from './ast.js';
import { type MILDataType, type MILType, ScalarType, TensorType, TupleType } from './types.js';

/** Represents the `Builder` class. */
export class Builder {
  /** Property `program` of `Builder`. */
  private program = new Program();
  /** Property `currentFunction` of `Builder`. */
  private currentFunction: MILFunction | null = null;
  /** Property `currentBlock` of `Builder`. */
  private currentBlock: Block | null = null;
  /** Property `varCounter` of `Builder`. */
  private varCounter = 0;

  /**
   * Executes `createProgram` on `Builder`.
   * @returns Result of `createProgram`.
   */
  createProgram(): Program {
    return this.program;
  }

  /**
   * Executes `createFunction` on `Builder`.
   * @param name Parameter `name`.
   * @param inputs Parameter `inputs`.
   * @param outputs Parameter `outputs`.
   * @returns Result of `createFunction`.
   */
  createFunction(name: string, inputs: Var[], outputs: Var[]): MILFunction {
    const fn = new MILFunction(name, inputs, outputs);
    this.program.addFunction(fn);
    this.currentFunction = fn;
    return fn;
  }

  /**
   * Executes `createBlock` on `Builder`.
   * @param name Parameter `name`.
   * @returns Result of `createBlock`.
   */
  createBlock(name: string): Block {
    if (!this.currentFunction) throw new Onnx9000Error('No active function');
    const block = new Block(name);
    this.currentFunction.addBlock(block);
    this.currentBlock = block;
    return block;
  }

  /**
   * Executes `setBlock` on `Builder`.
   * @param block Parameter `block`.
   * @returns Result of `setBlock`.
   */
  setBlock(block: Block) {
    this.currentBlock = block;
  }

  // Type system helpers
  /**
   * Executes `tensor` on `Builder`.
   * @param dtype Parameter `dtype`.
   * @param shape Parameter `shape`.
   * @returns Result of `tensor`.
   */
  tensor(dtype: MILDataType, shape: (number | string)[]): TensorType {
    return new TensorType(dtype, shape);
  }

  /**
   * Executes `scalar` on `Builder`.
   * @param dtype Parameter `dtype`.
   * @returns Result of `scalar`.
   */
  scalar(dtype: MILDataType): ScalarType {
    return new ScalarType(dtype);
  }

  /**
   * Executes `tuple` on `Builder`.
   * @param elements Parameter `elements`.
   * @returns Result of `tuple`.
   */
  tuple(elements: MILType[]): TupleType {
    return new TupleType(elements);
  }

  // Variable management
  /**
   * Executes `createVar` on `Builder`.
   * @param name Parameter `name`.
   * @param type Parameter `type`.
   * @returns Result of `createVar`.
   */
  createVar(name: string | null, type: MILType): Var {
    const varName = name || `%${this.varCounter++}`;
    return new Var(varName, type);
  }

  // Operations
  /**
   * Executes `addOp` on `Builder`.
   * @param opType Parameter `opType`.
   * @param inputs Parameter `inputs`.
   * @param outputs Parameter `outputs`.
   * @param attributes Parameter `attributes`.
   * @returns Result of `addOp`.
   */
  addOp(
    opType: string,
    inputs: Record<string, Var | Var[]>,
    outputs: Var[],
    attributes: Record<string, ReturnType<typeof JSON.parse>> = {},
  ): Operation {
    if (!this.currentBlock) throw new Onnx9000Error('No active block');
    const op = new Operation(opType, inputs, outputs, attributes);
    this.currentBlock.addOperation(op);
    return op;
  }

  // Common MIL operations
  /**
   * Executes `add` on `Builder`.
   * @param x Parameter `x`.
   * @param y Parameter `y`.
   * @param name Parameter `name`.
   * @returns Result of `add`.
   */
  add(x: Var, y: Var, name: string | null = null): Var {
    // Basic shape inference: assume same shape
    const outVar = this.createVar(name, x.type);
    this.addOp('add', { x, y }, [outVar]);
    return outVar;
  }

  /**
   * Executes `sub` on `Builder`.
   * @param x Parameter `x`.
   * @param y Parameter `y`.
   * @param name Parameter `name`.
   * @returns Result of `sub`.
   */
  sub(x: Var, y: Var, name: string | null = null): Var {
    const outVar = this.createVar(name, x.type);
    this.addOp('sub', { x, y }, [outVar]);
    return outVar;
  }

  /**
   * Executes `mul` on `Builder`.
   * @param x Parameter `x`.
   * @param y Parameter `y`.
   * @param name Parameter `name`.
   * @returns Result of `mul`.
   */
  mul(x: Var, y: Var, name: string | null = null): Var {
    const outVar = this.createVar(name, x.type);
    this.addOp('mul', { x, y }, [outVar]);
    return outVar;
  }

  /**
   * Executes `relu` on `Builder`.
   * @param x Parameter `x`.
   * @param name Parameter `name`.
   * @returns Result of `relu`.
   */
  relu(x: Var, name: string | null = null): Var {
    const outVar = this.createVar(name, x.type);
    this.addOp('relu', { x }, [outVar]);
    return outVar;
  }
}
