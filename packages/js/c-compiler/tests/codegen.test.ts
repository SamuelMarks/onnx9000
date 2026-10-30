import { Graph, Node } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { CFamilyCodegen, PythonFamilyCodegen } from '../src/codegen.js';

describe('codegen', () => {
  it('should codegen C', () => {
    const cg = new CFamilyCodegen();
    const graph = new Graph('test');
    graph.nodes.push(new Node('Add', [], []));
    const code = cg.visit(graph);
    expect(code).toContain('void forward_test()');
    expect(code).toContain('op_add');
  });

  it('should codegen Python', () => {
    const cg = new PythonFamilyCodegen();
    const graph = new Graph('test');
    graph.nodes.push(new Node('Add', [], []));
    const code = cg.visit(graph);
    expect(code).toContain('def forward_test');
    expect(code).toContain('add()');
  });
});
