import { describe, it, expect } from 'vitest';
import { Graph, Node, ValueInfo } from '@onnx9000/core';
import { generateTriton } from '../src/triton/ast.js';

describe('Triton Compiler', () => {
  it('should generate @triton.jit function with basic arguments', () => {
    const g = new Graph('test_kernel');
    g.inputs = [new ValueInfo('x', [1024], 'float32')];
    g.outputs = [new ValueInfo('y', [1024], 'float32')];

    const code = generateTriton(g);
    expect(code).toContain('import triton');
    expect(code).toContain('@triton.jit');
    expect(code).toContain('def test_kernel(');
    expect(code).toContain('x,');
    expect(code).toContain('y,');
    expect(code).toContain('BLOCK_M: tl.constexpr');
  });

  it('should generate tl.load with mask for 1D tensors', () => {
    const g = new Graph('add_kernel');
    g.inputs = [new ValueInfo('x', [1024], 'float32'), new ValueInfo('y', [1024], 'float32')];
    g.outputs = [new ValueInfo('z', [1024], 'float32')];
    g.addNode(new Node('Add', ['x', 'y'], ['z']));

    const code = generateTriton(g);
    expect(code).toContain('x_tile = tl.load(x_ptrs, mask=mask_m, other=0.0)');
    expect(code).toContain('z_var = x_tile + y_tile');
  });

  it('should generate code for various arithmetic ops', () => {
    const g = new Graph('math_kernel');
    g.inputs = [new ValueInfo('x', [1024], 'float32'), new ValueInfo('y', [1024], 'float32')];
    g.outputs = [new ValueInfo('z', [1024], 'float32')];
    g.addNode(new Node('Mul', ['x', 'y'], ['m']));
    g.addNode(new Node('Exp', ['m'], ['e']));
    g.addNode(new Node('Relu', ['e'], ['z']));

    const code = generateTriton(g);
    expect(code).toContain('m = x_tile * y_tile');
    expect(code).toContain('e = tl.exp(m)');
    expect(code).toContain('z_var = tl.maximum(e, 0.0)');
  });

  it('should generate code for reductions', () => {
    const g = new Graph('reduce_kernel');
    g.inputs = [new ValueInfo('x', [1024, 1024], 'float32')];
    g.outputs = [new ValueInfo('y', [1024], 'float32')];
    g.addNode(new Node('ReduceSum', ['x'], ['y']));

    const code = generateTriton(g);
    expect(code).toContain('y_var = tl.sum(x_tile, axis=0)');
  });

  it('should generate fused activation code', () => {
    const g = new Graph('act_kernel');
    g.inputs = [new ValueInfo('x', [1024], 'float32')];
    g.outputs = [new ValueInfo('y', [1024], 'float32')];
    g.addNode(new Node('LeakyRelu', ['x'], ['l'], { alpha: 0.1 }));
    g.addNode(new Node('Sigmoid', ['l'], ['y']));

    const code = generateTriton(g);
    expect(code).toContain('l = tl.where(x_tile > 0, x_tile, x_tile * 0.1)');
    expect(code).toContain('y_var = 1.0 / (1.0 + tl.exp(-l))');
  });

  it('should generate code for Where and Abs', () => {
    const g = new Graph('cond_kernel');
    g.inputs = [
      new ValueInfo('c', [1024], 'bool'),
      new ValueInfo('x', [1024], 'float32'),
      new ValueInfo('y', [1024], 'float32'),
    ];
    g.outputs = [new ValueInfo('z', [1024], 'float32')];
    g.addNode(new Node('Where', ['c', 'x', 'y'], ['w']));
    g.addNode(new Node('Abs', ['w'], ['z']));

    const code = generateTriton(g);
    expect(code).toContain('w = tl.where(c_tile, x_tile, y_tile)');
    expect(code).toContain('z_var = tl.abs(w)');
  });

  it('should generate code for Sin, Cos, Pow', () => {
    const g = new Graph('trig_kernel');
    g.inputs = [new ValueInfo('x', [1024], 'float32')];
    g.outputs = [new ValueInfo('y', [1024], 'float32')];
    g.addNode(new Node('Sin', ['x'], ['s']));
    g.addNode(new Node('Cos', ['s'], ['c']));
    g.addNode(new Node('Pow', ['c', 'x'], ['y']));

    const code = generateTriton(g);
    expect(code).toContain('s = tl.sin(x_tile)');
    expect(code).toContain('c = tl.cos(s)');
    expect(code).toContain('y_var = tl.math.pow(c, x_tile)');
  });

  it('should generate code for ArgMax/Min', () => {
    const g = new Graph('arg_kernel');
    g.inputs = [new ValueInfo('x', [1024], 'float32')];
    g.outputs = [new ValueInfo('y', [1], 'int64')];
    g.addNode(new Node('ArgMax', ['x'], ['y']));

    const code = generateTriton(g);
    expect(code).toContain('y_var = tl.argmax(x_tile, axis=0)');
  });

  it('should generate LayerNormalization code', () => {
    const g = new Graph('ln_kernel');
    g.inputs = [new ValueInfo('x', [1024, 1024], 'float32')];
    g.outputs = [new ValueInfo('y', [1024, 1024], 'float32')];
    g.addNode(new Node('LayerNormalization', ['x'], ['y']));

    const code = generateTriton(g);
    expect(code).toContain('y_var_mean = tl.sum(x_tile, axis=0) / BLOCK_N');
    expect(code).toContain('y_var_rsqrt = tl.math.rsqrt(y_var_var + 1e-5)');
  });

  it('should generate a launcher function', () => {
    const g = new Graph('launcher_test');
    g.inputs = [new ValueInfo('x', [1024], 'float32')];
    g.outputs = [new ValueInfo('y', [1024], 'float32')];

    const code = generateTriton(g);
    expect(code).toContain('def launcher_test_launcher(');
    expect(code).toContain('y = torch.empty_like(x)');
    expect(code).toContain('grid = lambda META:');
    expect(code).toContain('launcher_test[grid](');
  });

  it('should generate tl.dot for MatMul with loop', () => {
    const g = new Graph('matmul_kernel');
    g.inputs = [
      new ValueInfo('A', [128, 128], 'float32'),
      new ValueInfo('B', [128, 128], 'float32'),
    ];
    g.outputs = [new ValueInfo('C', [128, 128], 'float32')];
    g.addNode(new Node('MatMul', ['A', 'B'], ['C']));

    const code = generateTriton(g);
    expect(code).toContain('C_var = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)');
    expect(code).toContain('for k in range(0, K, BLOCK_K):');
    expect(code).toContain('C_var += tl.dot(a_tile, b_tile, allow_tf32=True)');
  });
});
