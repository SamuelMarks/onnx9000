import { Graph, Node, ValueInfo } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { generateTriton, TritonAST } from '../src/triton/ast.js';

describe('triton ast', () => {
  it('should generate code', () => {
    const graph = new Graph('test');
    graph.inputs.push(new ValueInfo('A', [10], 'float32'), new ValueInfo('B', [10], 'float32'));
    graph.outputs.push('C' as never as ValueInfo);
    graph.nodes.push(new Node('Add', ['A', 'B'], ['C'], {}));

    const code = generateTriton(graph);
    expect(code).toContain('@triton.jit');
    expect(code).toContain('def test');
    expect(code).toContain('C_var = A_tile + B_tile');
  });

  it('should handle AST properly', () => {
    const ast = new TritonAST();
    ast.pushLine('a');
    ast.indent();
    ast.pushLine('b');
    ast.dedent();
    expect(ast.getCode()).toBe('a\n    b');
  });
});
