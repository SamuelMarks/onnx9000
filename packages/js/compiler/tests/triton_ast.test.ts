import { describe, expect, it } from 'vitest';
import { generateTriton, TritonAST } from '../src/triton/ast.js';

describe('triton ast', () => {
  it('should generate code', () => {
    const graph: any = {
      name: 'test',
      inputs: [
        { name: 'A', shape: [10] },
        { name: 'B', shape: [10] },
      ],
      outputs: ['C'],
      nodes: [{ opType: 'Add', inputs: ['A', 'B'], outputs: ['C'], attributes: {} }],
    };

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
