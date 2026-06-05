import { describe, it, expect } from 'vitest';
import { MXNetGenerator } from '../src/mmdnn/mxnet/generator.js';

describe('MXNetGenerator', () => {
  it('should generate code', () => {
    const gen = new MXNetGenerator({
      name: 'test',
      inputs: [],
      outputs: [],
      tensors: {},
      nodes: [],
      valueInfo: [],
    } as any);
    const code = gen.generate();
    expect(code).toContain('import mxnet');
  });
});
