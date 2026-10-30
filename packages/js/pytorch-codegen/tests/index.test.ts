import { describe, expect, it } from 'vitest';
import { Pytorchcodegen } from '../src/index.js';

describe('Pytorchcodegen', () => {
  it('should generate PyTorch nn.Module code for Relu, Add, MatMul, and empty graphs', () => {
    const gen = new Pytorchcodegen();
    expect(gen.run()).toBe('[pytorch-codegen] processed');

    const code = gen.generateModule('MyNet', [
      { opType: 'Add', inputs: ['x', 'x'], output: 'h1' },
      { opType: 'MatMul', inputs: ['h1', 'x'], output: 'h2' },
      { opType: 'Relu', inputs: [], output: 'h3' },
      { opType: 'Add', inputs: [], output: 'h4' },
      { opType: 'MatMul', inputs: [], output: 'h5' },
    ]);
    expect(code).toContain('class MyNet(nn.Module):');
    expect(code).toContain('h1 = x + x');
    expect(code).toContain('h2 = torch.matmul(h1, x)');
    expect(code).toContain('return h5');

    expect(gen.generateModule('EmptyNet', [])).toContain('return x');
  });
});
