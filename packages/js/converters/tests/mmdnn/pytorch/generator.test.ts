import { describe, it, expect } from 'vitest';
import { PyTorchGenerator } from '../src/mmdnn/pytorch/generator.js';

describe('PyTorchGenerator', () => {
  it('should generate', () => {
    const gen = new PyTorchGenerator({
      name: 'test',
      inputs: [],
      outputs: [],
      tensors: {},
      nodes: [],
      valueInfo: [],
    } as any);
    const code = gen.generate();
    expect(code).toContain('import torch');
  });
});
