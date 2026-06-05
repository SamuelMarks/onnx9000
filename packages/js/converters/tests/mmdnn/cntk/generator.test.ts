import { describe, it, expect } from 'vitest';
import { CNTKGenerator } from '../src/mmdnn/cntk/generator.js';

describe('CNTKGenerator', () => {
  it('should generate', () => {
    const gen = new CNTKGenerator({
      name: 'test',
      inputs: [],
      outputs: [],
      tensors: {},
      nodes: [],
      valueInfo: [],
    } as any);
    const code = gen.generate();
    expect(code).toContain('import cntk');
    expect(code).toContain('def create_test');
  });
});
