import { Graph } from '@onnx9000/core';
import { describe, expect, it, vi } from 'vitest';
import { ONNXToMILConverter } from '../src/converter.js';

vi.mock('../src/mil/genai.js', () => ({
  detectAndMapGenAITopologies: vi.fn(),
}));
vi.mock('../src/mil/linter.js', () => ({ lintMILProgram: vi.fn() }));
vi.mock('../src/mil/batching.js', () => ({
  implementDynamicBatching: vi.fn(),
}));

describe('ONNXToMILConverter', () => {
  it('should convert graph', () => {
    const g = new Graph('test');
    g.inputs.push({ name: 'in', shape: [1], dtype: 'float32' });
    g.outputs.push({ name: 'out', shape: [1], dtype: 'float32' });
    g.nodes.push({
      opType: 'Relu',
      inputs: ['in'],
      outputs: ['out'],
      attributes: {},
    } as any);

    const conv = new ONNXToMILConverter(g);
    const prog = conv.convert();
    expect(prog).toBeDefined();
  });
});
