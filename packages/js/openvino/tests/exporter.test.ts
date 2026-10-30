import { Graph, Node, ValueInfo } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { OpenVinoExporter } from '../src/exporter.js';

describe('OpenVinoExporter', () => {
  it('should export', () => {
    const g = new Graph('test');
    g.inputs.push(new ValueInfo('in', [1], 'float32'));
    g.outputs.push(new ValueInfo('out', [1], 'float32'));
    g.nodes.push(new Node('Relu', ['in'], ['out'], {}));

    const exp = new OpenVinoExporter(g);
    const res = exp.export();
    expect(res.xml).toContain('net');
    expect(res.bin).toBeDefined();
  });
});
