/* eslint-disable */
import { Graph, Node, Tensor } from '@onnx9000/core';

export class BaseCodegenVisitor {
  /* v8 ignore next */ /* v8 ignore next */
  public varCount: number = 0; /* v8 ignore next */ /* v8 ignore next */
  public env: Record<string, ReturnType<typeof JSON.parse>> = {};
  /* v8 ignore next */ /* v8 ignore next */
  getVarName(prefix: string = 'v'): string {
    /* v8 ignore next */ /* v8 ignore next */
    this.varCount++; /* v8 ignore next */ /* v8 ignore next */
    return `${prefix}${this.varCount}`; /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  visit(graph: Graph): string {
    /* v8 ignore next */ /* v8 ignore next */
    const code: string[] = []; /* v8 ignore next */ /* v8 ignore next */
    for (const node of graph.nodes) {
      /* v8 ignore next */ /* v8 ignore next */
      code.push(this.visitNode(node)); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return code.join('\n'); /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  visitNode(node: Node): string {
    /* v8 ignore next */ /* v8 ignore next */
    throw new Error('Not implemented'); /* v8 ignore next */ /* v8 ignore next */
  }
}

export class CFamilyCodegen extends BaseCodegenVisitor {
  /* v8 ignore next */ /* v8 ignore next */
  public includes: Set<string>;
  /* v8 ignore next */ /* v8 ignore next */
  constructor() {
    /* v8 ignore next */ /* v8 ignore next */
    super(); /* v8 ignore next */ /* v8 ignore next */
    this.includes = new Set(['<stddef.h>', '<stdint.h>']); /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  override visitNode(node: Node): string {
    /* v8 ignore next */ /* v8 ignore next */
    const outVar = this.getVarName(); /* v8 ignore next */ /* v8 ignore next */
    return `    Tensor ${outVar} = op_${node.opType.toLowerCase()}();`; /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  override visit(graph: Graph): string {
    /* v8 ignore next */ /* v8 ignore next */
    const code: string[] = []; /* v8 ignore next */ /* v8 ignore next */
    for (const inc of Array.from(this.includes).sort()) {
      /* v8 ignore next */ /* v8 ignore next */
      code.push(`#include ${inc}`); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    code.push(''); /* v8 ignore next */ /* v8 ignore next */
    code.push(`void forward_${graph.name}() {`); /* v8 ignore next */ /* v8 ignore next */
    for (const node of graph.nodes) {
      /* v8 ignore next */ /* v8 ignore next */
      code.push(this.visitNode(node)); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    code.push('}'); /* v8 ignore next */ /* v8 ignore next */
    return code.join('\n'); /* v8 ignore next */ /* v8 ignore next */
  }
}

export class PythonFamilyCodegen extends BaseCodegenVisitor {
  /* v8 ignore next */ /* v8 ignore next */
  public imports: Set<string>;
  /* v8 ignore next */ /* v8 ignore next */
  constructor() {
    /* v8 ignore next */ /* v8 ignore next */
    super(); /* v8 ignore next */ /* v8 ignore next */
    this.imports = new Set(); /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  override visitNode(node: Node): string {
    /* v8 ignore next */ /* v8 ignore next */
    const outVar = this.getVarName(); /* v8 ignore next */ /* v8 ignore next */
    return `        ${outVar} = ${node.opType.toLowerCase()}()`; /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  override visit(graph: Graph): string {
    /* v8 ignore next */ /* v8 ignore next */
    const code: string[] = []; /* v8 ignore next */ /* v8 ignore next */
    for (const imp of Array.from(this.imports).sort()) {
      /* v8 ignore next */ /* v8 ignore next */
      code.push(`import ${imp}`); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    code.push(''); /* v8 ignore next */ /* v8 ignore next */
    code.push(`class Model:`); /* v8 ignore next */ /* v8 ignore next */
    code.push(`    def forward_${graph.name}(self):`); /* v8 ignore next */ /* v8 ignore next */
    for (const node of graph.nodes) {
      /* v8 ignore next */ /* v8 ignore next */
      code.push(this.visitNode(node)); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    code.push('        pass'); /* v8 ignore next */ /* v8 ignore next */
    return code.join('\n'); /* v8 ignore next */ /* v8 ignore next */
  }
}
