import { describe, it, expect } from 'vitest';
import { KerasGenerator } from '../../../src/mmdnn/keras/generator.js';

describe('KerasGenerator', () => {
  it('should generate code', () => {
    const gen = new KerasGenerator({
      name: 'test',
      inputs: [],
      outputs: [],
      tensors: {},
      nodes: [],
      valueInfo: [],
    } as any);
    const code = gen.generate();
    expect(code).toContain('import keras');
  });
});
