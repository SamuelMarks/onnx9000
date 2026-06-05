import { Graph, Node, Tensor } from '@onnx9000/core';

export class BaseCodegenVisitor {
  public varCount: number = 0;
  public env: Record<string, ReturnType<typeof JSON.parse>> = {};

  getVarName(prefix: string = 'v'): string {
    this.varCount++;
    return `${prefix}${this.varCount}`;
  }

  visit(graph: Graph): string {
    const code: string[] = [];
    for (const node of graph.nodes) {
      code.push(this.visitNode(node));
    }
    return code.join('\n');
  }

  visitNode(node: Node): string {
    throw new Error('Not implemented');
  }
}

export class CFamilyCodegen extends BaseCodegenVisitor {
  public includes: Set<string>;

  constructor() {
    super();
    this.includes = new Set(['<stddef.h>', '<stdint.h>']);
  }

  override visitNode(node: Node): string {
    const outVar = this.getVarName();
    return `    Tensor ${outVar} = op_${node.opType.toLowerCase()}();`;
  }

  override visit(graph: Graph): string {
    const code: string[] = [];
    for (const inc of Array.from(this.includes).sort()) {
      code.push(`#include ${inc}`);
    }
    code.push('');
    code.push(`void forward_${graph.name}() {`);
    for (const node of graph.nodes) {
      code.push(this.visitNode(node));
    }
    code.push('}');
    return code.join('\n');
  }
}

export class PythonFamilyCodegen extends BaseCodegenVisitor {
  public imports: Set<string>;

  constructor() {
    super();
    this.imports = new Set();
  }

  override visitNode(node: Node): string {
    const outVar = this.getVarName();
    return `        ${outVar} = ${node.opType.toLowerCase()}()`;
  }

  override visit(graph: Graph): string {
    const code: string[] = [];
    for (const imp of Array.from(this.imports).sort()) {
      code.push(`import ${imp}`);
    }
    code.push('');
    code.push(`class Model:`);
    code.push(`    def forward_${graph.name}(self):`);
    for (const node of graph.nodes) {
      code.push(this.visitNode(node));
    }
    code.push('        pass');
    return code.join('\n');
  }
}
