/**
 * @fileoverview printer.ts
 * Provides printer functionality for the coreml package.
 */
import type { Block, MILFunction, Operation, Program } from './ast.js';

/** Represents the `MILPrinter` class. */
export class MILPrinter {
  /** Property `indentLevel` of `MILPrinter`. */
  private indentLevel = 0;

  /**
   * Executes `indent` on `MILPrinter`.
   * @returns Result of `indent`.
   */
  private indent(): string {
    return '  '.repeat(this.indentLevel);
  }

  /**
   * Executes `printProgram` on `MILPrinter`.
   * @param prog Parameter `prog`.
   * @returns Result of `printProgram`.
   */
  printProgram(prog: Program): string {
    let result = '';
    for (const fn of Object.values(prog.functions)) {
      result += `${this.printFunction(fn)}\n`;
    }
    return result;
  }

  /**
   * Executes `printFunction` on `MILPrinter`.
   * @param fn Parameter `fn`.
   * @returns Result of `printFunction`.
   */
  printFunction(fn: MILFunction): string {
    let result = `${this.indent()}func @${fn.name}(`;
    result += fn.inputs.map((i) => `%${i.name}: ${i.type.toString()}`).join(', ');
    result += ') -> (';
    result += fn.outputs.map((o) => o.type.toString()).join(', ');
    result += ') {\n';

    this.indentLevel++;
    for (const block of Object.values(fn.blocks)) {
      result += this.printBlock(block);
    }
    this.indentLevel--;

    result += `${this.indent()}}\n`;
    return result;
  }

  /**
   * Executes `printBlock` on `MILPrinter`.
   * @param block Parameter `block`.
   * @returns Result of `printBlock`.
   */
  printBlock(block: Block): string {
    let result = `${this.indent()}${block.name}:\n`;
    this.indentLevel++;
    for (const op of block.operations) {
      result += `${this.printOperation(op)}\n`;
    }

    if (block.outputs.length > 0) {
      result += `${this.indent()}return ${block.outputs.map((o) => `%${o.name}`).join(', ')};\n`;
    }
    this.indentLevel--;
    return result;
  }

  /**
   * Executes `printOperation` on `MILPrinter`.
   * @param op Parameter `op`.
   * @returns Result of `printOperation`.
   */
  printOperation(op: Operation): string {
    const outs = op.outputs.map((o) => `%${o.name}`).join(', ');

    const ins = Object.entries(op.inputs)
      .map(([k, v]) => {
        const vals = Array.isArray(v)
          ? `[${v.map((vi) => `%${vi.name}`).join(', ')}]`
          : `%${v.name}`;
        return `${k}=${vals}`;
      })
      .join(', ');

    const attrs = Object.keys(op.attributes).length > 0 ? `, ${JSON.stringify(op.attributes)}` : '';

    return `${this.indent()}${outs} = ${op.opType}(${ins}${attrs})`;
  }
}
