import { Graph, Node, ValueInfo } from '@onnx9000/core';
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
    g.inputs.push(new ValueInfo('in', [1], 'float32'));
    g.outputs.push(new ValueInfo('out', [1], 'float32'));
    g.nodes.push(new Node('Relu', ['in'], ['out'], {}));

    const conv = new ONNXToMILConverter(g);
    const prog = conv.convert();
    expect(prog).toBeDefined();
  });
});
