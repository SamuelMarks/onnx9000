/**
 * @fileoverview codegen.ts
 * Provides codegen functionality for the c-compiler package.
 */

import type { Graph, Node } from '@onnx9000/core';
import { Onnx9000Error } from '@onnx9000/core';

/** Represents the `BaseCodegenVisitor` class. */
export class BaseCodegenVisitor {
  /** Property `varCount` of `BaseCodegenVisitor`. */
  public varCount: number = 0;
  /** Property `env` of `BaseCodegenVisitor`. */
  public env: Record<string, ReturnType<typeof JSON.parse>> = {};

  /**
   * Executes `getVarName` on `BaseCodegenVisitor`.
   * @param prefix Parameter `prefix`.
   * @returns Result of `getVarName`.
   */
  getVarName(prefix: string = 'v'): string {
    this.varCount++;
    return `${prefix}${this.varCount}`;
  }

  /**
   * Executes `visit` on `BaseCodegenVisitor`.
   * @param graph Parameter `graph`.
   * @returns Result of `visit`.
   */
  visit(graph: Graph): string {
    const code: string[] = [];
    for (const node of graph.nodes) {
      code.push(this.visitNode(node));
    }
    return code.join('\n');
  }

  /**
   * Executes `visitNode` on `BaseCodegenVisitor`.
   * @param _node Parameter `_node`.
   * @returns Result of `visitNode`.
   */
  visitNode(_node: Node): string {
    throw new Onnx9000Error('Not implemented');
  }
}

/** Represents the `CFamilyCodegen` class. */
export class CFamilyCodegen extends BaseCodegenVisitor {
  /** Property `includes` of `CFamilyCodegen`. */
  public includes: Set<string>;

  /** Initializes a new `CFamilyCodegen` instance. */
  constructor() {
    super();
    this.includes = new Set(['<stddef.h>', '<stdint.h>']);
  }

  /**
   * Executes `visitNode` on `CFamilyCodegen`.
   * @param node Parameter `node`.
   * @returns Result of `visitNode`.
   */
  override visitNode(node: Node): string {
    const outVar = this.getVarName();
    return `    Tensor ${outVar} = op_${node.opType.toLowerCase()}();`;
  }

  /**
   * Executes `visit` on `CFamilyCodegen`.
   * @param graph Parameter `graph`.
   * @returns Result of `visit`.
   */
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

/** Represents the `PythonFamilyCodegen` class. */
export class PythonFamilyCodegen extends BaseCodegenVisitor {
  /** Property `imports` of `PythonFamilyCodegen`. */
  public imports: Set<string>;

  /** Initializes a new `PythonFamilyCodegen` instance. */
  constructor() {
    super();
    this.imports = new Set();
  }

  /**
   * Executes `visitNode` on `PythonFamilyCodegen`.
   * @param node Parameter `node`.
   * @returns Result of `visitNode`.
   */
  override visitNode(node: Node): string {
    const outVar = this.getVarName();
    return `        ${outVar} = ${node.opType.toLowerCase()}()`;
  }

  /**
   * Executes `visit` on `PythonFamilyCodegen`.
   * @param graph Parameter `graph`.
   * @returns Result of `visit`.
   */
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
