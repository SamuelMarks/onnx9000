import { describe, it, expect } from 'vitest';
import { CaffeGenerator } from '../src/mmdnn/caffe/generator.js';

describe('CaffeGenerator', () => {
  it('should generate caffe code', () => {
    const g: any = { name: 'test', inputs: [], outputs: [], tensors: {}, nodes: [], valueInfo: [] };
    const gen = new CaffeGenerator(g);
    expect(gen.generate()).toContain('name: "test"');
  });
});
