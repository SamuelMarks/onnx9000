/* eslint-disable */
import { Program, Function, Block, Operation, Var } from './ast.js';

export class MILPrinter {
  private indentLevel = 0;

  private indent(): string {
    return '  '.repeat(this.indentLevel);
  }

  printProgram(prog: Program): string {
    let result = '';
    for (const name in prog.functions) {
      result += this.printFunction(prog.functions[name]!) + '\n';
    }
    return result;
  }

  printFunction(fn: Function): string {
    let result = `${this.indent()}func @${fn.name}(`;
    result += fn.inputs.map((i) => `%${i.name}: ${i.type.toString()}`).join(', ');
    result += ') -> (';
    result += fn.outputs.map((o) => o.type.toString()).join(', ');
    result += ') {\n';

    this.indentLevel++;
    for (const blockName in fn.blocks) {
      result += this.printBlock(fn.blocks[blockName]!);
    }
    this.indentLevel--;

    result += `${this.indent()}}\n`;
    return result;
  }

  printBlock(block: Block): string {
    let result = `${this.indent()}${block.name}:\n`;
    this.indentLevel++;
    for (const op of block.operations) {
      result += this.printOperation(op) + '\n';
    }

    if (block.outputs.length > 0) {
      result += `${this.indent()}return ${block.outputs.map((o) => `%${o.name}`).join(', ')};\n`;
    }
    this.indentLevel--;
    return result;
  }

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
