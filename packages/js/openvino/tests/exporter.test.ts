import { Graph } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { OpenVinoExporter } from '../src/exporter.js';

describe('OpenVinoExporter', () => {
  it('should export', () => {
    const g = new Graph('test');
    g.inputs.push({ name: 'in', shape: [1], dtype: 'float32' } as any);
    g.outputs.push({ name: 'out', shape: [1], dtype: 'float32' } as any);
    g.nodes.push({
      opType: 'Relu',
      inputs: ['in'],
      outputs: ['out'],
      attributes: {},
    } as any);

    const exp = new OpenVinoExporter(g);
    const res = exp.export();
    expect(res.xml).toContain('net');
    expect(res.bin).toBeDefined();
  });
});
