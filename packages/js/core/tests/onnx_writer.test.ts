import { describe, it, expect } from 'vitest';
import { serializeModelProto } from '../src/parser/onnx_writer.js';
import { Graph } from '../ir/graph.js';

describe('onnx_writer', () => {
  it('should serialize model proto', () => {
    const g = new Graph('test');
    g.nodes.push({
      opType: 'Add',
      inputs: ['a', 'b'],
      outputs: ['c'],
      attributes: {},
      name: 'n1',
    } as any);
    g.inputs.push({ name: 'a', shape: [1], dtype: 'float32' } as any);
    g.outputs.push({ name: 'c', shape: [1], dtype: 'float32' } as any);

    const bytes = serializeModelProto(g);
    expect(bytes).toBeDefined();
    expect(bytes.length).toBeGreaterThan(0);
  });
});
